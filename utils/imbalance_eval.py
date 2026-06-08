"""
Evaluasi penanganan class imbalance pada Dataset_TehHijau lewat *resampling data*
(over/under/both) dan membandingkannya dengan *adjusted model* (class_weight /
sample_weight / auto_class_weights 'balanced').

Split CV:
  * Seluruh project memakai StratifiedKFold (hanya menjaga proporsi kelas, TIDAK
    menghormati grup Sampling_ID/Chop_ID). ⚠️ Untuk Dataset_TehHijau yang ber-grup
    ini berarti replika dari satu sampel bisa tersebar ke train+val sekaligus,
    sehingga metrik CV cenderung optimistis (lihat memory
    grouped-data-leakage-teh-hijau). Ini sesuai permintaan eksplisit.
  * Resampling (SMOTE/under/SMOTEENN) HANYA dijalankan pada fold training. Ini
    dijamin oleh imblearn.Pipeline: sampler aktif saat .fit(), dilewati saat
    .predict() terhadap fold validasi. Jadi fold validasi tetap distribusi asli.

Dipakai oleh notebook di notebooks/handle_imbalance/.
"""

import time
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, precision_score, recall_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn import FunctionSampler

# Best params per model (selaras notebooks/3.4.2_..._newhpobalanced.ipynb)
# Terapkan nilai baru ini untuk menekan overfitting
BEST_PARAMS = {
    "SVC Linear":      {"C": 0.1}, # Turunkan C untuk memperlebar margin
    "SVC Sigmoid":     {"C": 0.1, "gamma": "scale", "coef0": 0.0},
    "SVC Poly": {
        "C": 0.01,           # Diturunkan ekstrem dari 0.1 untuk memperluas margin toleransi
        "degree": 2,         # Turunkan kompleksitas polynomial dari 3 ke 2
        "gamma": "scale"
    },
    "SVC RBF": {
        "C": 0.05,           # Diturunkan drastis agar tidak membentuk pulau-pulau overfit
        "gamma": "scale"
    },
    "XGBoost GBTree": {
        "n_estimators": 100,       # Batasi jumlah pohon secara drastis (sebelumnya 200/500)
        "learning_rate": 0.03,
        "max_depth": 2,            # Paksa menjadi decision stump (kedalaman 2)
        "min_child_weight": 15,    # Butuh minimal 15 sampel untuk membuat cabang baru
        "reg_alpha": 10,           # Regularisasi L1 yang kuat untuk memangkas fitur tidak penting
        "reg_lambda": 100,         # Regularisasi L2 super tinggi untuk meratakan bobot daun
        "gamma": 5                 # Penalti minimal loss reduction untuk setiap split baru
    },
    "XGBoost Dart": {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 15,
        "reg_alpha": 10,
        "reg_lambda": 100,
        "gamma": 5,
        "rate_drop": 0.4,          # Naikkan drop rate pohon sebesar 40% agar drop-out bekerja maksimal
        "skip_drop": 0.5
    },
    "CatBoost": {
        "iterations": 80,          # Turunkan iterasi secara agresif (sebelumnya 150/200)
        "depth": 3,                # Batasi kedalaman maksimal pohon di angka 3
        "learning_rate": 0.03,
        "l2_leaf_reg": 30,         # Naikkan regularisasi L2 secara masif (sebelumnya 10)
        "random_strength": 5,      # Tambahkan noise pada struktur pohon untuk mencegah hafalan
        "bagging_temperature": 1.0
    }
}

MODEL_ORDER = list(BEST_PARAMS.keys())


def make_cv_splitter(cv_strategy="stratified", n_splits=5, state=42):
    """Bangun splitter CV.

    Seluruh project kini memakai StratifiedKFold (hanya menjaga proporsi kelas,
    TIDAK menghormati grup Sampling_ID/Chop_ID). Argumen `groups` yang diteruskan
    ke .split() diterima tapi diabaikan oleh StratifiedKFold.

    `cv_strategy` dipertahankan demi kompatibilitas pemanggil lama: nilai 'group'
    maupun 'stratified' sama-sama menghasilkan StratifiedKFold (keputusan project,
    lihat memory cv-stratifiedkfold-projectwide).

    'stratified_group' adalah opt-in eksplisit: StratifiedGroupKFold yang MENJAGA
    grup `groups` (mis. Sampling_ID) -> replika satu sampel tak tersebar ke
    train+val. Hanya dipakai notebook yang sengaja memintanya; default project
    tetap StratifiedKFold.
    """
    if cv_strategy in ("group", "stratified"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)
    if cv_strategy == "stratified_group":
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=state)
    raise ValueError(
        f"Unknown cv_strategy: {cv_strategy!r} "
        "(pakai 'group', 'stratified', atau 'stratified_group')")


class XGBBalancedWrapper(BaseEstimator, ClassifierMixin):
    """XGBClassifier yang otomatis compute_sample_weight('balanced') tiap .fit()."""
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        self.model_ = XGBClassifier(**self.xgb_params)
        sw = compute_sample_weight("balanced", y) if sample_weight is None else sample_weight
        self.model_.fit(X, y, sample_weight=sw)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_estimator(model_name, params, balanced, xgb_device="cuda",
                   cat_task_type="GPU", probability=True):
    """Bangun classifier. balanced=True -> pakai bobot kelas (mode adjusted);
    balanced=False -> tanpa bobot, biar efek murni resampling yang terlihat.
    probability=False mempercepat SVC (lewati kalibrasi Platt) saat learning curve
    yang hanya butuh .predict()."""
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    cw = "balanced" if balanced else None

    if model_name == "SVC Linear":
        return SVC(kernel="linear", probability=probability, random_state=42,
                   decision_function_shape="ovr", class_weight=cw, **params)
    if model_name == "SVC Poly":
        return SVC(kernel="poly", probability=probability, random_state=42,
                   decision_function_shape="ovr", class_weight=cw, **params)
    if model_name == "SVC RBF":
        return SVC(kernel="rbf", probability=probability, random_state=42,
                   decision_function_shape="ovr", class_weight=cw, **params)
    if model_name == "SVC Sigmoid":
        return SVC(kernel="sigmoid", probability=probability, random_state=42,
                   decision_function_shape="ovr", class_weight=cw, **params)
    if model_name in ("XGBoost GBTree", "XGBoost Dart"):
        booster = "gbtree" if model_name == "XGBoost GBTree" else "dart"
        common = dict(booster=booster, objective="multi:softprob",
                      random_state=42, device=xgb_device, **params)
        return XGBBalancedWrapper(**common) if balanced else XGBClassifier(**common)
    if model_name == "CatBoost":
        return CatBoostClassifier(
            loss_function="MultiClassOneVsAll", eval_metric="Accuracy", verbose=0,
            random_seed=42, task_type=cat_task_type, devices="0",
            auto_class_weights="Balanced" if balanced else None, **params)
    raise ValueError(f"Unknown model_name: {model_name}")


def _build_pipeline(model_name, params, balanced, resampler, n_optimal,
                    xgb_device, cat_task_type, probability=True):
    steps = [("scaler", StandardScaler()), ("pca", PCA(n_components=n_optimal))]
    if resampler is not None:
        steps.append(("resampler", resampler))  # hanya aktif saat fit (fold train)
    steps.append(("clf", make_estimator(model_name, params, balanced,
                                        xgb_device, cat_task_type, probability)))
    return ImbPipeline(steps)


def make_resample_pipeline(model_name, params, resampler, n_optimal,
                           xgb_device="cuda", cat_task_type="GPU", probability=False):
    """Pipeline mode resample (classifier TANPA bobot kelas) untuk learning curve.
    Clone-able oleh sklearn.learning_curve; resampler hanya aktif di subset training
    tiap fold -> tidak bocor. probability=False default biar SVC cepat."""
    return _build_pipeline(model_name, params, balanced=False, resampler=resampler,
                           n_optimal=n_optimal, xgb_device=xgb_device,
                           cat_task_type=cat_task_type, probability=probability)


def make_custom_resampler(targets, class_names, random_state=42, smote_k=5,
                          over_method="smote", under_method="random",
                          borderline_kind="borderline-1", borderline_m=10):
    """Sampler kustom dengan target jumlah per-kelas, dihitung **per fold**.

    `targets`: dict {nama_kelas: target}, di mana target boleh:
      * int            -> jumlah absolut, atau
      * nama kelas lain -> samakan dengan jumlah kelas itu DI FOLD TRAINING saat itu
                           (mis. {'B': 'C', 'E': 'C'} -> B & E disamakan ke jumlah C).
    Kelas yang TIDAK disebut di `targets` -> tidak diubah.

    Untuk tiap kelas: target < jumlah sekarang -> di-undersample; target > jumlah
    sekarang -> di-oversample. Keduanya dibungkus dalam satu FunctionSampler supaya
    bisa dipakai sebagai langkah tunggal di imblearn.Pipeline (CV) maupun di
    learning_curve. Target dihitung ulang tiap fold via Counter(y), sehingga
    "samakan ke C/D" tetap benar walau jumlah C/D beda antar fold.

    `over_method`  : 'smote' (default) | 'borderline' | 'smoteenn' | 'smotetomek'
        -> SMOTE, BorderlineSMOTE, atau metode *combine* (SMOTE + pembersihan ENN/Tomek).
    `under_method` : 'random' (default) | 'tomek'     -> RandomUnderSampler atau TomekLinks.

    ⚠️ TomekLinks adalah metode *cleaning*, bukan undersampler bertarget: ia hanya
    membuang pasangan Tomek (majority-minority bertetangga) dari kelas yang dipilih
    dan TIDAK bisa menurunkan ke jumlah target tertentu. Jadi saat
    under_method='tomek', nilai target untuk kelas yang di-undersample HANYA dipakai
    untuk memilih kelas mana yang dibersihkan — jumlah akhirnya ditentukan data.

    ⚠️ over_method 'smoteenn'/'smotetomek': bagian SMOTE menaikkan kelas `over` ke
    target, lalu langkah pembersihan (ENN/Tomek) membuang sampel batas/noise dari
    SEMUA kelas. Jadi jumlah akhir tiap kelas bisa sedikit di bawah target (ENN lebih
    agresif dari Tomek), dan kelas lain pun ikut terpangkas — "samakan ke C" jadi
    perkiraan, bukan eksak. Borderline-SMOTE hanya mensintesis dari sampel di batas;
    kelas yang sudah terpisah rapi ('safe') bisa TIDAK bertambah sama sekali.
    """
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek

    name_to_label = {n: i for i, n in enumerate(class_names)}
    spec = {name_to_label[k]: v for k, v in targets.items()}

    def _resample(X, y):
        cnt = Counter(np.asarray(y).tolist())

        def resolve(v):
            return int(cnt[name_to_label[v]]) if isinstance(v, str) else int(v)

        under = {lbl: resolve(v) for lbl, v in spec.items() if resolve(v) < cnt[lbl]}
        over = {lbl: resolve(v) for lbl, v in spec.items() if resolve(v) > cnt[lbl]}

        X_r, y_r = X, y
        if under:
            if under_method == "tomek":
                # TomekLinks tak menerima target count -> hanya pilih kelas yg dibersihkan.
                X_r, y_r = TomekLinks(
                    sampling_strategy=list(under.keys())
                ).fit_resample(X_r, y_r)
            else:
                X_r, y_r = RandomUnderSampler(
                    sampling_strategy=under, random_state=random_state
                ).fit_resample(X_r, y_r)
        if over:
            if over_method == "borderline":
                X_r, y_r = BorderlineSMOTE(
                    sampling_strategy=over, random_state=random_state,
                    k_neighbors=smote_k, m_neighbors=borderline_m, kind=borderline_kind
                ).fit_resample(X_r, y_r)
            elif over_method in ("smoteenn", "smotetomek"):
                # SMOTE ke target lalu bersihkan batas. Inner SMOTE eksplisit supaya
                # smote_k dihormati; sampling_strategy=over membatasi over ke kelas target.
                smote = SMOTE(sampling_strategy=over, random_state=random_state,
                              k_neighbors=smote_k)
                Combine = SMOTEENN if over_method == "smoteenn" else SMOTETomek
                X_r, y_r = Combine(
                    sampling_strategy=over, random_state=random_state, smote=smote
                ).fit_resample(X_r, y_r)
            else:
                X_r, y_r = SMOTE(
                    sampling_strategy=over, random_state=random_state, k_neighbors=smote_k
                ).fit_resample(X_r, y_r)
        return X_r, y_r

    return FunctionSampler(func=_resample, validate=False)


def resample_for_viz(X, y, groups, resampler, n_optimal,
                     n_splits=5, state=42, fold_index=0, cv_strategy="group"):
    """Replikasi apa yang terjadi pada SATU fold training saat CV: fit
    StandardScaler+PCA pada fold train, lalu jalankan resampler di ruang PCA.
    Dipakai untuk visualisasi komposisi 'sebelum vs sesudah' sampling.

    Catatan: jumlah baris per kelas tidak dipengaruhi PCA (label y utuh); PCA
    hanya relevan untuk sebaran fitur (scatter). Mengembalikan koordinat PCA
    sebelum & sesudah supaya bisa di-scatter sekaligus.

    `cv_strategy`: dipertahankan demi kompatibilitas; 'group'/'stratified'
    sama-sama StratifiedKFold (grup diabaikan) -> lihat make_cv_splitter.
    """
    skf = make_cv_splitter(cv_strategy, n_splits=n_splits, state=state)
    tr, _ = list(skf.split(X, y, groups))[fold_index]
    X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
    y_tr = np.asarray(y)[tr]

    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=n_optimal).fit(scaler.transform(X_tr))
    X_before = pca.transform(scaler.transform(X_tr))

    X_after, y_after = resampler.fit_resample(X_before, y_tr)
    n_orig = len(y_tr)  # imblearn over/combine: baris asli dulu, lalu sintetis
    return {
        "X_before": np.asarray(X_before), "y_before": y_tr,
        "X_after": np.asarray(X_after), "y_after": np.asarray(y_after),
        "n_orig": n_orig,
    }


def evaluate_cv(X, y, groups, model_name, params, n_optimal,
                mode, resampler=None, labels=None,
                n_splits=5, state=42, cv_strategy="group",
                xgb_device="cuda", cat_task_type="GPU", log=print):
    """
    Jalankan CV untuk satu model.

    mode = 'adjusted'  -> tanpa resampler, classifier pakai bobot kelas balanced.
    mode = 'resample'  -> pakai `resampler` di fold train, classifier tanpa bobot.

    cv_strategy: dipertahankan demi kompatibilitas; 'group'/'stratified' sama-sama
                 StratifiedKFold (grup Sampling_ID/Chop_ID diabaikan, lihat
                 make_cv_splitter).

    Return dict berisi metrik agregat (mean) + prediksi out-of-fold untuk
    confusion matrix.
    """
    assert mode in ("adjusted", "resample")
    balanced = (mode == "adjusted")
    use_resampler = resampler if mode == "resample" else None
    if labels is None:
        labels = np.unique(y)

    skf = make_cv_splitter(cv_strategy, n_splits=n_splits, state=state)

    accs, bals, f1w, f1m, rocs, pras, precs, recs = [], [], [], [], [], [], [], []
    y_true_all, y_pred_all = [], []
    t0 = time.perf_counter()

    for fold, (tr, va) in enumerate(skf.split(X, y, groups), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        pipe = _build_pipeline(model_name, params, balanced, use_resampler,
                               n_optimal, xgb_device, cat_task_type)
        pipe.fit(X_tr, y_tr)

        y_pred = np.ravel(pipe.predict(X_va))
        y_prob = pipe.predict_proba(X_va)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # normalisasi (CatBoost)

        y_true_all.extend(y_va)
        y_pred_all.extend(y_pred)

        Yb = label_binarize(y_va, classes=labels)
        accs.append(accuracy_score(y_va, y_pred))
        bals.append(balanced_accuracy_score(y_va, y_pred))
        f1w.append(f1_score(y_va, y_pred, average="weighted", zero_division=0))
        f1m.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        rocs.append(roc_auc_score(y_va, y_prob, average="weighted", multi_class="ovr"))
        pras.append(average_precision_score(Yb, y_prob, average="weighted"))
        precs.append(precision_score(y_va, y_pred, average="weighted", zero_division=0))
        recs.append(recall_score(y_va, y_pred, average="weighted", zero_division=0))

        log(f"    F{fold} -> Acc={accs[-1]:.4f} | BalAcc={bals[-1]:.4f} | "
            f"MacroF1={f1m[-1]:.4f} | AUROC={rocs[-1]:.4f} | PRAUC={pras[-1]:.4f}")

    exec_time = time.perf_counter() - t0
    mcc = matthews_corrcoef(y_true_all, y_pred_all)

    # Nilai per-fold tiap metrik -> dipakai untuk mean, std, dan rincian per fold.
    fold_metrics = {
        "balanced_accuracy": bals,
        "accuracy": accs,
        "macro_f1": f1m,
        "weighted_f1": f1w,
        "precision": precs,
        "recall": recs,
        "roc_auc": rocs,
        "pr_auc": pras,
    }

    res = {
        "model": model_name,
        "mode": mode,
        "cv_strategy": cv_strategy,
        "params": params,
        # mean tiap metrik (kompatibel dengan pemakaian lama)
        **{m: float(np.mean(v)) for m, v in fold_metrics.items()},
        # std antar-fold tiap metrik
        **{f"{m}_std": float(np.std(v)) for m, v in fold_metrics.items()},
        # nilai per fold tiap metrik (list panjang n_splits)
        **{f"{m}_folds": [float(x) for x in v] for m, v in fold_metrics.items()},
        "mcc": float(mcc),
        "execution_time": exec_time,
        "y_true": list(map(int, y_true_all)),
        "y_pred": list(map(int, y_pred_all)),
    }
    log(f"  OK [{model_name} | {mode}] BalAcc={res['balanced_accuracy']:.4f} "
        f"MacroF1={res['macro_f1']:.4f} AUROC={res['roc_auc']:.4f} "
        f"MCC={res['mcc']:.4f} ({exec_time:.1f}s)")
    return res
