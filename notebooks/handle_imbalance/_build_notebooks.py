"""
Generator 3 notebook penanganan class imbalance lewat RESAMPLING DATA:
  oversampling_smote.ipynb     -> SMOTE
  undersampling_random.ipynb   -> RandomUnderSampler
  combine_smoteenn.ipynb       -> SMOTEENN

Tiap notebook membandingkan strategi resampling (data-level) vs adjusted model
(class_weight/sample_weight/auto_class_weights 'balanced') pada 7 model identik
dengan notebooks/3.4.2_..._newhpobalanced.ipynb. Logika CV ada di
utils/imbalance_eval.py (resampling hanya di fold train -> anti-bocor).

Jalankan: python notebooks/handle_imbalance/_build_notebooks.py
"""
import json


def _src(s):
    # nbformat menyimpan source sebagai list baris dengan newline trailing
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(s)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(s)}


SCENARIOS = {
    "oversampling_smote": {
        "title": "Oversampling (SMOTE) vs Adjusted Model",
        "blurb": (
            "**Skenario 1 — Oversampling.** Kelas minoritas (A, B) digandakan "
            "secara sintetis pakai **SMOTE** sampai seimbang dengan kelas mayoritas "
            "(E). SMOTE hanya dijalankan pada **fold training** lewat `imblearn.Pipeline` "
            "supaya fold validasi tetap distribusi asli (anti-bocor)."
        ),
        "import": "from imblearn.over_sampling import SMOTE",
        "resampler": "SMOTE(random_state=42, k_neighbors=5)",
        "tag": "oversampling",
        "warn": (
            "> ⚠️ **Catatan runtime:** oversampling membesarkan training set "
            "(~5×4649 ≈ 23k baris/fold). SVC (`probability=True`) jadi **sangat "
            "lambat** O(N²). Kalau hanya butuh sebagian model, komentari entri "
            "yang tak perlu di `MODELS`."
        ),
    },
    "undersampling_random": {
        "title": "Undersampling (Random) vs Adjusted Model",
        "blurb": (
            "**Skenario 2 — Undersampling.** Kelas mayoritas dikurangi acak pakai "
            "**RandomUnderSampler** sampai sebanyak kelas minoritas. Training set jadi "
            "kecil & cepat, tapi banyak informasi kelas mayoritas terbuang. "
            "Resampling hanya pada **fold training** (anti-bocor)."
        ),
        "import": "from imblearn.under_sampling import RandomUnderSampler",
        "resampler": "RandomUnderSampler(random_state=42)",
        "tag": "undersampling",
        "warn": (
            "> ℹ️ Undersampling membuat training set kecil (≈ jumlah kelas minoritas × "
            "n_kelas). Cepat dijalankan, cocok untuk iterasi."
        ),
    },
    "custom_sampling": {
        "title": "Custom Sampling (target per-kelas) vs Adjusted Model",
        "blurb": (
            "**Skenario 3 — Custom (keduanya, target per-kelas).** Atur sendiri "
            "kelas mana di-resample ke berapa lewat `CUSTOM_TARGETS`. Kelas dengan "
            "target di bawah jumlahnya akan di-**undersample** (RandomUnderSampler), "
            "di atas jumlahnya di-**oversample** (SMOTE); kelas yang tak disebut "
            "**tidak diubah**. Default: **B & E disamakan ke jumlah C** (A, C, D "
            "tetap). Target dihitung **per fold** sehingga \"samakan ke C/D\" tetap "
            "benar walau jumlah C/D beda antar fold. Resampling hanya pada **fold "
            "training** (anti-bocor)."
        ),
        "import": "from utils.imbalance_eval import make_custom_resampler",
        "resampler": "make_custom_resampler(CUSTOM_TARGETS, class_names)",
        "tag": "custom",
        "custom": True,
        "warn": (
            "> ℹ️ Edit `CUSTOM_TARGETS` di sel Konfigurasi untuk coba-coba "
            "(mis. `{'B': 'C', 'E': 'C'}`, atau angka absolut `{'B': 2000, 'E': 2000}`). "
            "Kalau ada kelas dioversample besar, SVC `probability=True` bisa lambat."
        ),
    },
}


def build(scn_key, scn):
    cells = []

    # 0. Title
    cells.append(md(
        f"# Handle Class Imbalance — {scn['title']}\n\n"
        f"{scn['blurb']}\n\n"
        "Dataset `Dataset_TehHijau.csv` **ber-grup**: 10.409 baris tapi hanya 274 "
        "`Sampling_ID` (≈38 pembacaan sensor mirip per sampel). Maka CV memakai "
        "`StratifiedKFold(groups=Sampling_ID)` agar pembacaan sampel yang sama "
        "tak bocor ke train+val. Metrik andalan untuk imbalance: **balanced "
        "accuracy** & **macro-F1** (bukan accuracy).\n\n"
        f"{scn['warn']}"
    ))

    # 1. install
    cells.append(code(
        "!python -m pip install -q numpy pandas scikit-learn imbalanced-learn \\\n"
        "    xgboost catboost matplotlib seaborn"
    ))

    # 2. imports + project root
    cells.append(code(
        "import sys, os\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "cwd = os.path.abspath(os.getcwd())\n"
        "project_root = cwd.split('codes')[0] + 'codes'\n"
        "sys.path.append(os.path.abspath(project_root))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.float_format', '{:.4f}'.format)"
    ))

    # 3. reload helper (biar edit utils kebaca tanpa restart kernel)
    cells.append(code(
        "import importlib\n\n"
        "def reload_package(package_name):\n"
        "    for name in sorted([n for n in sys.modules if n.startswith(package_name)], reverse=True):\n"
        "        importlib.reload(sys.modules[name])\n\n"
        "reload_package('utils')"
    ))

    # 4. config
    if scn.get("custom"):
        resampler_block = (
            "# Target sampling kustom per-kelas — EDIT untuk coba-coba.\n"
            "#   nilai = int (jumlah absolut) ATAU nama kelas lain (samakan jumlahnya).\n"
            "#   kelas yang TIDAK disebut = tidak diubah.\n"
            "# Default: B & E disamakan ke jumlah C; A, C, D tetap.\n"
            "CUSTOM_TARGETS = {'B': 'C', 'E': 'C'}\n"
            "RESAMPLER = None   # dibangun setelah class_names tersedia (sel di bawah)\n"
        )
    else:
        resampler_block = (
            "# Resampler skenario ini (dipakai HANYA pada fold training oleh evaluate_cv)\n"
            f"RESAMPLER = {scn['resampler']}\n"
        )
    cells.append(md("## 1. Konfigurasi"))
    cells.append(code(
        "from imblearn.over_sampling import SMOTE  # noqa\n"
        "from imblearn.under_sampling import RandomUnderSampler  # noqa\n"
        "from imblearn.combine import SMOTEENN  # noqa\n"
        f"{scn['import']}\n"
        "from utils.imbalance_eval import evaluate_cv, BEST_PARAMS, MODEL_ORDER\n\n"
        "dataset_path = os.path.join(project_root, 'dataset', 'Dataset_TehHijau.csv')\n\n"
        "feature_cols = [\n"
        "    'MQ3', 'TGS822', 'TGS2602', 'MQ5', 'MQ138', 'TGS2620',\n"
        "    'TGS813', 'TGS2600', 'TGS2611', 'TGS2603', 'Humidity', 'Celsius',\n"
        "]\n"
        "target_col = 'Kategori'\n"
        "group_col  = 'Sampling_ID'   # grup anti-bocor (lihat memory grouped-data-leakage)\n\n"
        f"{resampler_block}"
        f"STRATEGY  = '{scn['tag']}'\n\n"
        "# Backend hardware — samakan dengan notebook 3.4.2 (GPU). Ganti ke\n"
        "# 'cpu'/'CPU' kalau tidak ada GPU.\n"
        "XGB_DEVICE   = 'cuda'\n"
        "CAT_TASK_TYPE = 'GPU'\n\n"
        "# Model yang dibandingkan (komentari yang tak perlu untuk menghemat waktu)\n"
        "MODELS = list(MODEL_ORDER)\n"
        "print('Models:', MODELS)\n\n"
        "# Hasil disimpan relatif terhadap folder notebook ini\n"
        "OUT_DIR = os.path.join(cwd, 'results', STRATEGY)\n"
        "os.makedirs(OUT_DIR, exist_ok=True)\n"
        "print('Output:', OUT_DIR)"
    ))

    # 5. load data
    cells.append(md("## 2. Muat Data & Distribusi Kelas"))
    cells.append(code(
        "data = pd.read_csv(dataset_path)\n"
        "print('shape:', data.shape, '| n Sampling_ID:', data[group_col].nunique())\n"
        "data.head()"
    ))
    cells.append(code(
        "dist = data[target_col].value_counts().sort_index()\n"
        "print('Distribusi kelas (baris):')\n"
        "print(dist)\n"
        "print('\\nRasio imbalance (maks/min): {:.1f}x'.format(dist.max() / dist.min()))\n\n"
        "ax = dist.plot(kind='bar', color='steelblue')\n"
        "ax.set_xlabel('Kategori'); ax.set_ylabel('Count')\n"
        "ax.set_title('Distribusi Kelas — Dataset Teh Hijau (imbalanced)')\n"
        "plt.tight_layout(); plt.show()"
    ))

    # 6. X y groups + encode
    cells.append(code(
        "from sklearn.preprocessing import LabelEncoder\n\n"
        "X = data[feature_cols]\n"
        "label_encoder = LabelEncoder()\n"
        "y = label_encoder.fit_transform(data[target_col])\n"
        "groups = data[group_col].values\n"
        "labels = np.unique(y)\n"
        "class_names = list(label_encoder.classes_)\n"
        "print('X:', X.shape, '| classes:', class_names)"
    ))

    if scn.get("custom"):
        cells.append(code(
            "# Bangun resampler kustom dari CUSTOM_TARGETS (butuh class_names).\n"
            "# Target dihitung per fold; B & E -> jumlah C, A/C/D tetap (default).\n"
            "RESAMPLER = make_custom_resampler(CUSTOM_TARGETS, class_names)\n"
            "print('Resampler kustom siap | CUSTOM_TARGETS =', CUSTOM_TARGETS)"
        ))

    # 7. PCA variance
    cells.append(md("## 3. PCA — jumlah komponen optimal (95% varians)"))
    cells.append(code(
        "from sklearn.decomposition import PCA\n\n"
        "_pca = PCA(n_components=X.shape[1]).fit(X)\n"
        "cumvar = np.cumsum(_pca.explained_variance_ratio_)\n"
        "n_optimal = int(np.argmax(cumvar >= 0.95) + 1)\n\n"
        "plt.figure(figsize=(7, 4))\n"
        "plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o')\n"
        "plt.axhline(0.95, color='gray', ls='--')\n"
        "plt.axvline(n_optimal, color='red', ls='--')\n"
        "plt.xlabel('n komponen'); plt.ylabel('Cumulative explained variance')\n"
        "plt.title(f'PCA — n_optimal = {n_optimal}'); plt.grid(True); plt.tight_layout(); plt.show()\n"
        "print('n_optimal =', n_optimal)"
    ))

    # 7b. komposisi sebelum vs sesudah sampling
    cells.append(md(
        "## 4. Komposisi Dataset — Sebelum vs Sesudah Sampling\n\n"
        f"Resampling (`{scn['resampler']}`) dijalankan pada **1 fold training** "
        "(persis seperti yang terjadi di CV: fit `StandardScaler`+`PCA` pada fold "
        "train, lalu resample di ruang PCA).\n\n"
        "- **Kiri (histogram):** jumlah baris per kelas sebelum vs sesudah. Ini "
        "**tidak** terpengaruh PCA — label kelas tetap utuh, PCA hanya mengubah "
        "fitur.\n"
        "- **Tengah & kanan (scatter PC1–PC2):** sebaran fitur di ruang PCA sebelum "
        "vs sesudah; di sinilah efek PCA terlihat — titik sintetis (SMOTE) muncul "
        "di antara titik minoritas nyata.\n\n"
        "Ditampilkan untuk 1 fold sebagai ilustrasi; tiap fold di CV diresample "
        "dengan cara sama. Fold **validasi tidak diresample**."
    ))
    cells.append(code(
        "from collections import Counter\n"
        "from utils.imbalance_eval import resample_for_viz\n\n"
        "viz = resample_for_viz(X, y, groups, RESAMPLER, n_optimal, fold_index=0)\n"
        "cb, ca = Counter(viz['y_before']), Counter(viz['y_after'])\n"
        "comp = pd.DataFrame(\n"
        "    {'Sebelum': [cb.get(i, 0) for i in labels],\n"
        "     'Sesudah': [ca.get(i, 0) for i in labels]}, index=class_names)\n"
        "comp['Delta'] = comp['Sesudah'] - comp['Sebelum']\n"
        "print(f'Komposisi kelas pada 1 fold training (strategi={STRATEGY}):')\n"
        "print(comp)\n"
        "comp.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_composition.csv'))\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
        "# (a) histogram jumlah kelas sebelum vs sesudah\n"
        "comp[['Sebelum', 'Sesudah']].plot(\n"
        "    kind='bar', ax=axes[0], color={'Sebelum': 'lightgray', 'Sesudah': 'steelblue'})\n"
        "axes[0].set_title(f'Komposisi kelas — sebelum vs sesudah {STRATEGY}')\n"
        "axes[0].set_xlabel('Kategori'); axes[0].set_ylabel('Jumlah baris')\n"
        "axes[0].tick_params(axis='x', rotation=0)\n"
        "for c in axes[0].containers:\n"
        "    axes[0].bar_label(c, fontsize=8)\n\n"
        "# (b)(c) scatter PCA PC1-PC2 sebelum vs sesudah\n"
        "cmap = plt.cm.tab10\n"
        "for ax, (Xp, yp, ttl) in zip(axes[1:], [\n"
        "        (viz['X_before'], viz['y_before'], 'Sebelum'),\n"
        "        (viz['X_after'],  viz['y_after'],  'Sesudah')]):\n"
        "    for i, cn in enumerate(class_names):\n"
        "        m = yp == labels[i]\n"
        "        ax.scatter(Xp[m, 0], Xp[m, 1], s=8, alpha=0.4,\n"
        "                   color=cmap(i), label=f'{cn} (n={int(m.sum())})')\n"
        "    ax.set_title(f'PCA PC1-PC2 — {ttl} {STRATEGY}')\n"
        "    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\n"
        "    ax.legend(fontsize=7, markerscale=2)\n"
        "plt.tight_layout()\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_composition_before_after.png'),\n"
        "            dpi=130, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # 8. run evaluation
    cells.append(md(
        "## 5. Evaluasi: Resampling vs Adjusted Model\n\n"
        "Untuk tiap model dijalankan **dua kondisi** dengan split & params identik:\n\n"
        "- **adjusted** — tanpa resampling, classifier pakai bobot kelas `balanced` "
        "(seperti notebook 3.4.2).\n"
        f"- **resample** — pakai `{scn['resampler']}` di fold training, classifier "
        "**tanpa** bobot kelas (supaya efek murni resampling yang terukur).\n\n"
        "Resampling hanya pada fold training; fold validasi selalu distribusi asli."
    ))
    cells.append(code(
        "import time\n"
        "from pathlib import Path\n"
        "from datetime import datetime\n\n"
        "log_path = Path(OUT_DIR) / f'{STRATEGY}_run.log'\n"
        "log_path.write_text('', encoding='utf-8')\n\n"
        "def log(msg=''):\n"
        "    print(msg)\n"
        "    with open(log_path, 'a', encoding='utf-8') as f:\n"
        "        f.write(str(msg) + '\\n')\n\n"
        "log(f'Started: {datetime.now().isoformat(timespec=\"seconds\")} | strategy={STRATEGY}')\n\n"
        "common = dict(n_optimal=n_optimal, labels=labels,\n"
        "              xgb_device=XGB_DEVICE, cat_task_type=CAT_TASK_TYPE, log=log)\n\n"
        "all_results = []   # baris per (model, mode)\n"
        "for name in MODELS:\n"
        "    params = BEST_PARAMS[name]\n"
        "    for mode, resampler in [('adjusted', None), ('resample', RESAMPLER)]:\n"
        "        log(f'\\n=== {name} | {mode} ===')\n"
        "        try:\n"
        "            res = evaluate_cv(X, y, groups, name, params, mode=mode,\n"
        "                              resampler=resampler, **common)\n"
        "            all_results.append(res)\n"
        "        except Exception as e:\n"
        "            log(f'  FAILED [{name} | {mode}] -> {type(e).__name__}: {e}')\n\n"
        "log(f'\\nDone. {len(all_results)} (model x mode) berhasil.')"
    ))

    # 9. comparison table
    cells.append(md("## 6. Tabel Perbandingan"))
    cells.append(code(
        "metric_cols = ['balanced_accuracy', 'macro_f1', 'accuracy', 'weighted_f1',\n"
        "               'precision', 'recall', 'roc_auc', 'pr_auc', 'mcc', 'execution_time']\n"
        "rows = []\n"
        "for r in all_results:\n"
        "    rows.append({'Model': r['model'], 'Strategy': r['mode'],\n"
        "                 **{m: r[m] for m in metric_cols}})\n"
        "res_df = pd.DataFrame(rows)\n"
        "res_df.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_metrics_long.csv'), index=False)\n"
        "res_df.style.format({m: '{:.4f}' for m in metric_cols})"
    ))
    # 10. paired view + delta
    cells.append(code(
        "# Tampilan berpasangan: resample vs adjusted + selisih (positif = resampling lebih baik)\n"
        "key = ['balanced_accuracy', 'macro_f1', 'roc_auc', 'mcc']\n"
        "piv = res_df.pivot_table(index='Model', columns='Strategy', values=key)\n"
        "delta = pd.DataFrame(index=piv.index)\n"
        "for m in key:\n"
        "    if ('resample' in piv[m]) and ('adjusted' in piv[m]):\n"
        "        delta[f'{m}\\nadjusted'] = piv[(m, 'adjusted')]\n"
        "        delta[f'{m}\\nresample'] = piv[(m, 'resample')]\n"
        "        delta[f'{m}\\nDelta']    = piv[(m, 'resample')] - piv[(m, 'adjusted')]\n"
        "delta = delta.reindex([m for m in MODELS if m in delta.index])\n"
        "delta.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_paired_delta.csv'))\n"
        "def _hl(v):\n"
        "    if isinstance(v, float):\n"
        "        return 'color: green' if v > 0 else ('color: red' if v < 0 else '')\n"
        "    return ''\n"
        "_delta_cols = [c for c in delta.columns if c.endswith('Delta')]\n"
        "_sty = delta.style.format('{:.4f}')\n"
        "# pandas >=2.1 pakai Styler.map; versi lama pakai applymap\n"
        "(_sty.map if hasattr(_sty, 'map') else _sty.applymap)(_hl, subset=_delta_cols)"
    ))
    cells.append(code(
        "# Ringkasan: rata-rata selisih + berapa model yang membaik\n"
        "for m in key:\n"
        "    col = f'{m}\\nDelta'\n"
        "    if col in delta.columns:\n"
        "        d = delta[col]\n"
        "        better = (d > 0).sum(); worse = (d < 0).sum()\n"
        "        log(f'{m:<20s} mean delta={d.mean():+.4f} | resampling menang di {better}/{len(d)} model (kalah {worse})')"
    ))

    # 11. bar chart
    cells.append(md("## 7. Visual: Balanced Accuracy & Macro-F1 (adjusted vs resample)"))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "for ax, metric, title in zip(axes, ['balanced_accuracy', 'macro_f1'],\n"
        "                             ['Balanced Accuracy', 'Macro-F1']):\n"
        "    sub = res_df.pivot(index='Model', columns='Strategy', values=metric)\n"
        "    sub = sub.reindex([m for m in MODELS if m in sub.index])\n"
        "    sub = sub[[c for c in ['adjusted', 'resample'] if c in sub.columns]]\n"
        "    sub.plot(kind='bar', ax=ax, color={'adjusted': 'lightgray', 'resample': 'steelblue'})\n"
        "    ax.set_title(f'{title} — adjusted vs {STRATEGY}')\n"
        "    ax.set_ylabel(title); ax.set_ylim(0, 1.0)\n"
        "    ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_bar_compare.png'), dpi=130, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # 12. confusion matrices
    cells.append(md(
        "## 8. Confusion Matrix Out-of-Fold (perhatikan kelas minoritas B)\n\n"
        "Tiap baris = satu model; kiri = adjusted, kanan = resample. Prediksi "
        "out-of-fold mencerminkan generalisasi, bukan performa di data training."
    ))
    cells.append(code(
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report\n\n"
        "by_model = {}\n"
        "for r in all_results:\n"
        "    by_model.setdefault(r['model'], {})[r['mode']] = r\n"
        "ordered = [m for m in MODELS if m in by_model]\n\n"
        "nrows = len(ordered)\n"
        "fig, axes = plt.subplots(nrows, 2, figsize=(10, 4.2 * nrows))\n"
        "axes = np.atleast_2d(axes)\n"
        "for i, name in enumerate(ordered):\n"
        "    for j, mode in enumerate(['adjusted', 'resample']):\n"
        "        ax = axes[i, j]\n"
        "        r = by_model[name].get(mode)\n"
        "        if r is None:\n"
        "            ax.axis('off'); continue\n"
        "        cm = confusion_matrix(r['y_true'], r['y_pred'], labels=labels)\n"
        "        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(\n"
        "            ax=ax, cmap='Blues', colorbar=False, values_format='d')\n"
        "        ax.set_title(f\"{name} | {mode}\\nBalAcc={r['balanced_accuracy']:.3f} MacroF1={r['macro_f1']:.3f}\")\n"
        "        ax.tick_params(axis='x', rotation=45)\n"
        "fig.suptitle(f'Confusion Matrix OOF — adjusted vs {STRATEGY}', fontsize=14)\n"
        "fig.tight_layout()\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_confusion_matrices.png'), dpi=130, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(code(
        "# Classification report per kelas (fokus recall kelas minoritas B)\n"
        "for name in ordered:\n"
        "    for mode in ['adjusted', 'resample']:\n"
        "        r = by_model[name].get(mode)\n"
        "        if r is None: continue\n"
        "        log('=' * 64)\n"
        "        log(f'{name} | {mode}')\n"
        "        log(classification_report(r['y_true'], r['y_pred'], labels=labels,\n"
        "                                  target_names=class_names, zero_division=0))"
    ))

    # 12b. learning curve (overfit/underfit) untuk model + resampling
    cells.append(md(
        "## 9. Learning Curve — cek Overfit / Underfit\n\n"
        f"Train vs validation **balanced_accuracy** pada beberapa ukuran training "
        f"set untuk tiap model dalam mode **resample** (`{scn['resampler']}` di fold "
        "training, classifier **tanpa** bobot kelas — sesuai keputusan: kalau sudah "
        "resampling, model tak perlu di-`balanced` lagi).\n\n"
        "- `gap = train − val` besar (>0.15) -> **overfit**; val rendah (<0.5) -> "
        "**underfit**; gap kecil & val tinggi -> **well-fit**.\n"
        "- CV & grup sama (`StratifiedKFold` by Sampling_ID); resampling hanya "
        "di subset training tiap fold -> tidak bocor.\n\n"
        "> ⚠️ SVC pada skenario oversampling/SMOTEENN tetap berat. `probability=False` "
        "dipakai di sini (LC hanya butuh `.predict()`) supaya lebih cepat."
    ))
    cells.append(code(
        "from sklearn.model_selection import learning_curve, StratifiedKFold\n"
        "from utils.imbalance_eval import make_resample_pipeline\n\n"
        "LC_TRAIN_SIZES = np.linspace(0.2, 1.0, 5)   # 5 titik kurva\n"
        "LC_CV          = 5\n"
        "LC_SCORING     = 'balanced_accuracy'\n"
        "LC_STATE       = 42\n"
        "groups_arr = np.asarray(groups)\n\n"
        "lc_results = []\n"
        "for name in MODELS:\n"
        "    params = BEST_PARAMS[name]\n"
        "    log(f'  > learning curve: {name} ...')\n"
        "    try:\n"
        "        pipe = make_resample_pipeline(name, params, RESAMPLER, n_optimal,\n"
        "                                      xgb_device=XGB_DEVICE, cat_task_type=CAT_TASK_TYPE)\n"
        "        ts_abs, tr_sc, va_sc = learning_curve(\n"
        "            pipe, X, y, groups=groups_arr,\n"
        "            train_sizes=LC_TRAIN_SIZES,\n"
        "            cv=StratifiedKFold(n_splits=LC_CV, shuffle=True, random_state=LC_STATE),\n"
        "            scoring=LC_SCORING, n_jobs=1, shuffle=True, random_state=LC_STATE)\n"
        "        lc_results.append({'model': name, 'train_sizes': ts_abs,\n"
        "                           'train_mean': tr_sc.mean(1), 'train_std': tr_sc.std(1),\n"
        "                           'val_mean': va_sc.mean(1), 'val_std': va_sc.std(1)})\n"
        "    except Exception as e:\n"
        "        log(f'    FAILED [{name}] -> {type(e).__name__}: {e}')\n"
        "log(f'Learning curve selesai untuk {len(lc_results)} model.')"
    ))
    cells.append(code(
        "ncols = 3\n"
        "nrows = (len(lc_results) + ncols - 1) // ncols\n"
        "fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),\n"
        "                         constrained_layout=True)\n"
        "axes = np.array(axes).flatten()\n"
        "for ax, r in zip(axes, lc_results):\n"
        "    ts = r['train_sizes']\n"
        "    ax.plot(ts, r['train_mean'], 'o-', color='steelblue', lw=2, label='Train')\n"
        "    ax.fill_between(ts, r['train_mean'] - r['train_std'], r['train_mean'] + r['train_std'],\n"
        "                    alpha=0.2, color='steelblue')\n"
        "    ax.plot(ts, r['val_mean'], 's-', color='darkorange', lw=2, label='Val')\n"
        "    ax.fill_between(ts, r['val_mean'] - r['val_std'], r['val_mean'] + r['val_std'],\n"
        "                    alpha=0.2, color='darkorange')\n"
        "    gap = r['train_mean'][-1] - r['val_mean'][-1]\n"
        "    if   gap > 0.15:                              verdict = 'OVERFIT'\n"
        "    elif r['val_mean'][-1] < 0.5:                 verdict = 'UNDERFIT'\n"
        "    elif gap < 0.05 and r['val_mean'][-1] > 0.7:  verdict = 'Well-fit'\n"
        "    else:                                         verdict = 'Mild gap'\n"
        "    ax.set_title(f\"{r['model']}\\ngap={gap:+.3f} | {verdict}\", fontsize=11, fontweight='bold')\n"
        "    ax.set_xlabel('Training set size'); ax.set_ylabel(LC_SCORING)\n"
        "    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.4); ax.legend(fontsize=8)\n"
        "for ax in axes[len(lc_results):]:\n"
        "    ax.set_visible(False)\n"
        "fig.suptitle(f'Learning Curve — {STRATEGY} (model tanpa balanced) | scoring={LC_SCORING}',\n"
        "             fontsize=13, fontweight='bold')\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_learning_curve.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "log('\\n---- Ringkasan learning curve (training size terbesar) ----')\n"
        "log(f\"{'Model':<18s}{'Train':>8s}{'Val':>8s}{'Gap':>8s}\")\n"
        "for r in sorted(lc_results, key=lambda x: -x['val_mean'][-1]):\n"
        "    log(f\"{r['model']:<18s}{r['train_mean'][-1]:>8.4f}{r['val_mean'][-1]:>8.4f}\"\n"
        "        f\"{r['train_mean'][-1] - r['val_mean'][-1]:>+8.4f}\")"
    ))

    # 13. conclusion md
    cells.append(md(
        "## 10. Kesimpulan\n\n"
        "Bandingkan kolom **Delta** di tabel berpasangan (sel 6) dan recall kelas "
        "**B** di confusion matrix:\n\n"
        "- Delta `balanced_accuracy`/`macro_f1` **positif** -> resampling "
        f"(`{scn['tag']}`) mengungguli adjusted model untuk model tsb.\n"
        "- Perhatikan trade-off: resampling kadang menaikkan recall minoritas tapi "
        "menurunkan precision / accuracy keseluruhan.\n"
        "- **Learning curve (sel 9)** menunjukkan overfit/underfit tiap model di mode "
        "resampling (tanpa balanced). Gap train-val besar -> overfit; val rendah -> underfit.\n"
        "- Semua angka dari `StratifiedKFold(Sampling_ID)` + resampling khusus "
        "fold training, jadi sudah bebas kebocoran antar-sampel.\n\n"
        "File hasil tersimpan di `OUT_DIR` (CSV metrik, paired delta, PNG bar/CM/learning curve)."
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ───────────────────────────────────────────────────────────────────────────
# Notebook MULTI-TEKNIK: bandingkan beberapa resampler sekaligus (mis. hybrid)
# ───────────────────────────────────────────────────────────────────────────
MULTI_SCENARIOS = {
    "hybrid_sampling": {
        "title": "Hybrid Sampling (SMOTEENN vs SMOTETomek) vs Adjusted Model",
        "blurb": (
            "**Hybrid sampling** = oversampling (SMOTE) lalu pembersihan mayoritas. "
            "Dua varian dibandingkan: **SMOTEENN** (SMOTE + Edited Nearest Neighbours) "
            "dan **SMOTETomek** (SMOTE + Tomek Links). Keduanya diadu dengan baseline "
            "**adjusted** (class_weight balanced) pada 7 model identik. Resampling "
            "hanya pada **fold training** (anti-bocor)."
        ),
        "imports": "from imblearn.combine import SMOTEENN, SMOTETomek",
        "resamplers_block": (
            "RESAMPLERS = {\n"
            "    'SMOTEENN':   SMOTEENN(random_state=42),\n"
            "    'SMOTETomek': SMOTETomek(random_state=42),\n"
            "}\n"
        ),
        "tag": "hybrid",
        "warn": (
            "> ⚠️ **Runtime:** hybrid meng-oversample ke level mayoritas (~23k baris/"
            "fold) lalu cleaning -> SVC `probability=True` sangat lambat, dan tiap "
            "model dijalankan untuk **tiap** teknik. Komentari model/teknik yang tak "
            "perlu di `MODELS`/`RESAMPLERS`."
        ),
    },
}


def build_multi(scn):
    """Notebook yang membandingkan adjusted vs BEBERAPA resampler (RESAMPLERS dict)."""
    cells = []
    tag = scn["tag"]

    cells.append(md(
        f"# Handle Class Imbalance — {scn['title']}\n\n"
        f"{scn['blurb']}\n\n"
        "Dataset `Dataset_TehHijau.csv` **ber-grup**: 274 `Sampling_ID`. CV memakai "
        "`StratifiedKFold(groups=Sampling_ID)`; metrik andalan **balanced "
        "accuracy** & **macro-F1**.\n\n"
        f"{scn['warn']}"
    ))
    cells.append(code(
        "!python -m pip install -q numpy pandas scikit-learn imbalanced-learn \\\n"
        "    xgboost catboost matplotlib seaborn"
    ))
    cells.append(code(
        "import sys, os\nimport warnings\nwarnings.filterwarnings('ignore')\n\n"
        "cwd = os.path.abspath(os.getcwd())\n"
        "project_root = cwd.split('codes')[0] + 'codes'\n"
        "sys.path.append(os.path.abspath(project_root))\n\n"
        "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.float_format', '{:.4f}'.format)"
    ))
    cells.append(code(
        "import importlib\n\n"
        "def reload_package(package_name):\n"
        "    for name in sorted([n for n in sys.modules if n.startswith(package_name)], reverse=True):\n"
        "        importlib.reload(sys.modules[name])\n\n"
        "reload_package('utils')"
    ))

    # config
    cells.append(md("## 1. Konfigurasi"))
    cells.append(code(
        scn["imports"] + "\n"
        "from utils.imbalance_eval import (\n"
        "    evaluate_cv, resample_for_viz, make_resample_pipeline, BEST_PARAMS, MODEL_ORDER)\n\n"
        "dataset_path = os.path.join(project_root, 'dataset', 'Dataset_TehHijau.csv')\n\n"
        "feature_cols = [\n"
        "    'MQ3', 'TGS822', 'TGS2602', 'MQ5', 'MQ138', 'TGS2620',\n"
        "    'TGS813', 'TGS2600', 'TGS2611', 'TGS2603', 'Humidity', 'Celsius',\n"
        "]\n"
        "target_col = 'Kategori'\n"
        "group_col  = 'Sampling_ID'\n\n"
        "# Teknik resampling yang dibandingkan (urutan dipakai di tabel/plot).\n"
        "# Komentari yang tak perlu untuk menghemat waktu.\n"
        + scn["resamplers_block"] +
        "STRATEGY = '" + tag + "'\n\n"
        "XGB_DEVICE   = 'cuda'   # ganti 'cpu' kalau tak ada GPU\n"
        "CAT_TASK_TYPE = 'GPU'   # ganti 'CPU' kalau tak ada GPU\n\n"
        "MODELS = list(MODEL_ORDER)\n"
        "print('Models:', MODELS)\n"
        "print('Teknik:', list(RESAMPLERS.keys()))\n\n"
        "OUT_DIR = os.path.join(cwd, 'results', STRATEGY)\n"
        "os.makedirs(OUT_DIR, exist_ok=True)\n"
        "print('Output:', OUT_DIR)"
    ))

    # load + dist
    cells.append(md("## 2. Muat Data & Distribusi Kelas"))
    cells.append(code(
        "data = pd.read_csv(dataset_path)\n"
        "print('shape:', data.shape, '| n Sampling_ID:', data[group_col].nunique())\n"
        "data[target_col].value_counts().sort_index()"
    ))
    cells.append(code(
        "from sklearn.preprocessing import LabelEncoder\n\n"
        "X = data[feature_cols]\n"
        "label_encoder = LabelEncoder()\n"
        "y = label_encoder.fit_transform(data[target_col])\n"
        "groups = data[group_col].values\n"
        "labels = np.unique(y)\n"
        "class_names = list(label_encoder.classes_)\n"
        "print('X:', X.shape, '| classes:', class_names)"
    ))

    # PCA
    cells.append(md("## 3. PCA — jumlah komponen optimal (95% varians)"))
    cells.append(code(
        "from sklearn.decomposition import PCA\n\n"
        "_pca = PCA(n_components=X.shape[1]).fit(X)\n"
        "cumvar = np.cumsum(_pca.explained_variance_ratio_)\n"
        "n_optimal = int(np.argmax(cumvar >= 0.95) + 1)\n"
        "print('n_optimal =', n_optimal)"
    ))

    # composition (multi)
    cells.append(md(
        "## 4. Komposisi Dataset — Sebelum vs Sesudah (tiap teknik)\n\n"
        "Jumlah baris per kelas pada 1 fold training, untuk tiap teknik. Histogram "
        "tidak terpengaruh PCA; scatter PC1–PC2 menampilkan sebaran fitur."
    ))
    cells.append(code(
        "from collections import Counter\n\n"
        "comp = pd.DataFrame(index=class_names)\n"
        "before_xy = None\n"
        "scatters = []\n"
        "for tech, R in RESAMPLERS.items():\n"
        "    v = resample_for_viz(X, y, groups, R, n_optimal, fold_index=0)\n"
        "    if before_xy is None:\n"
        "        before_xy = (v['X_before'], v['y_before'])\n"
        "        comp['Sebelum'] = [Counter(v['y_before']).get(i, 0) for i in labels]\n"
        "    comp[tech] = [Counter(v['y_after']).get(i, 0) for i in labels]\n"
        "    scatters.append((tech, v['X_after'], v['y_after']))\n"
        "print('Komposisi kelas (1 fold training):'); print(comp)\n"
        "comp.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_composition.csv'))\n\n"
        "ax = comp.plot(kind='bar', figsize=(10, 5))\n"
        "ax.set_title(f'Komposisi kelas — Sebelum vs {STRATEGY}')\n"
        "ax.set_xlabel('Kategori'); ax.set_ylabel('Jumlah baris'); ax.tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_composition_bar.png'), dpi=130, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "cmap = plt.cm.tab10\n"
        "panels = [('Sebelum', before_xy[0], before_xy[1])] + [(t, Xa, ya) for (t, Xa, ya) in scatters]\n"
        "ncols = len(panels)\n"
        "fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))\n"
        "axes = np.atleast_1d(axes)\n"
        "for ax, (ttl, Xp, yp) in zip(axes, panels):\n"
        "    for i, cn in enumerate(class_names):\n"
        "        m = yp == labels[i]\n"
        "        ax.scatter(Xp[m, 0], Xp[m, 1], s=8, alpha=0.4, color=cmap(i), label=f'{cn} (n={int(m.sum())})')\n"
        "    ax.set_title(f'PCA PC1-PC2 — {ttl}'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\n"
        "    ax.legend(fontsize=7, markerscale=2)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_composition_scatter.png'), dpi=130, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # evaluation
    cells.append(md(
        "## 5. Evaluasi: Adjusted vs tiap teknik hybrid\n\n"
        "Tiap model dijalankan untuk **adjusted** (balanced, tanpa resampling) lalu "
        "**tiap teknik** (resampling di fold training, classifier tanpa bobot)."
    ))
    cells.append(code(
        "import time\nfrom pathlib import Path\nfrom datetime import datetime\n\n"
        "log_path = Path(OUT_DIR) / f'{STRATEGY}_run.log'\n"
        "log_path.write_text('', encoding='utf-8')\n\n"
        "def log(msg=''):\n"
        "    print(msg)\n"
        "    with open(log_path, 'a', encoding='utf-8') as f:\n"
        "        f.write(str(msg) + '\\n')\n\n"
        "log(f'Started: {datetime.now().isoformat(timespec=\"seconds\")} | strategy={STRATEGY}')\n\n"
        "common = dict(n_optimal=n_optimal, labels=labels,\n"
        "              xgb_device=XGB_DEVICE, cat_task_type=CAT_TASK_TYPE, log=log)\n"
        "STRATEGIES = ['adjusted'] + list(RESAMPLERS.keys())\n\n"
        "all_results = []\n"
        "for name in MODELS:\n"
        "    params = BEST_PARAMS[name]\n"
        "    for strat in STRATEGIES:\n"
        "        log(f'\\n=== {name} | {strat} ===')\n"
        "        try:\n"
        "            if strat == 'adjusted':\n"
        "                r = evaluate_cv(X, y, groups, name, params, mode='adjusted', **common)\n"
        "            else:\n"
        "                r = evaluate_cv(X, y, groups, name, params, mode='resample',\n"
        "                                resampler=RESAMPLERS[strat], **common)\n"
        "            r['strategy'] = strat\n"
        "            all_results.append(r)\n"
        "        except Exception as e:\n"
        "            log(f'  FAILED [{name} | {strat}] -> {type(e).__name__}: {e}')\n"
        "log(f'\\nDone. {len(all_results)} (model x strategi).')"
    ))

    # table
    cells.append(md("## 6. Tabel Perbandingan"))
    cells.append(code(
        "metric_cols = ['balanced_accuracy', 'macro_f1', 'accuracy', 'weighted_f1',\n"
        "               'precision', 'recall', 'roc_auc', 'pr_auc', 'mcc', 'execution_time']\n"
        "rows = [{'Model': r['model'], 'Strategy': r['strategy'],\n"
        "         **{m: r[m] for m in metric_cols}} for r in all_results]\n"
        "res_df = pd.DataFrame(rows)\n"
        "res_df.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_metrics_long.csv'), index=False)\n"
        "res_df.style.format({m: '{:.4f}' for m in metric_cols})"
    ))
    cells.append(code(
        "# Strategi terbaik per model (balanced_accuracy)\n"
        "best = (res_df.sort_values('balanced_accuracy', ascending=False)\n"
        "        .groupby('Model', sort=False).first()[['Strategy', 'balanced_accuracy', 'macro_f1', 'roc_auc', 'mcc']])\n"
        "best = best.reindex([m for m in MODELS if m in best.index])\n"
        "best.to_csv(os.path.join(OUT_DIR, f'{STRATEGY}_best_per_model.csv'))\n"
        "log('Strategi terbaik per model (balanced_accuracy):')\n"
        "for m, row in best.iterrows():\n"
        "    log(f\"  {m:<16s} -> {row['Strategy']:<12s} BalAcc={row['balanced_accuracy']:.4f} MacroF1={row['macro_f1']:.4f}\")\n"
        "best"
    ))

    # grouped bar
    cells.append(md("## 7. Visual: Balanced Accuracy & Macro-F1 per strategi"))
    cells.append(code(
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
        "for ax, metric, title in zip(axes, ['balanced_accuracy', 'macro_f1'],\n"
        "                             ['Balanced Accuracy', 'Macro-F1']):\n"
        "    sub = res_df.pivot(index='Model', columns='Strategy', values=metric)\n"
        "    sub = sub.reindex(index=[m for m in MODELS if m in sub.index],\n"
        "                      columns=[s for s in STRATEGIES if s in sub.columns])\n"
        "    sub.plot(kind='bar', ax=ax)\n"
        "    ax.set_title(f'{title} — adjusted vs {STRATEGY}'); ax.set_ylabel(title)\n"
        "    ax.set_ylim(0, 1.0); ax.tick_params(axis='x', rotation=45)\n"
        "    ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_bar_compare.png'), dpi=130, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # confusion matrices
    cells.append(md(
        "## 8. Confusion Matrix Out-of-Fold\n\n"
        "Kolom = strategi (adjusted + tiap teknik); baris = model."
    ))
    cells.append(code(
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report\n\n"
        "by_model = {}\n"
        "for r in all_results:\n"
        "    by_model.setdefault(r['model'], {})[r['strategy']] = r\n"
        "ordered = [m for m in MODELS if m in by_model]\n\n"
        "ncols = len(STRATEGIES); nrows = len(ordered)\n"
        "fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))\n"
        "axes = np.atleast_2d(axes)\n"
        "for i, name in enumerate(ordered):\n"
        "    for j, strat in enumerate(STRATEGIES):\n"
        "        ax = axes[i, j]; r = by_model[name].get(strat)\n"
        "        if r is None:\n"
        "            ax.axis('off'); continue\n"
        "        cm = confusion_matrix(r['y_true'], r['y_pred'], labels=labels)\n"
        "        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(\n"
        "            ax=ax, cmap='Blues', colorbar=False, values_format='d')\n"
        "        ax.set_title(f\"{name} | {strat}\\nBalAcc={r['balanced_accuracy']:.3f}\", fontsize=9)\n"
        "        ax.tick_params(axis='x', rotation=45)\n"
        "fig.suptitle(f'Confusion Matrix OOF — {STRATEGY}', fontsize=14)\n"
        "fig.tight_layout()\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_confusion_matrices.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(code(
        "for name in ordered:\n"
        "    for strat in STRATEGIES:\n"
        "        r = by_model[name].get(strat)\n"
        "        if r is None: continue\n"
        "        log('=' * 64); log(f'{name} | {strat}')\n"
        "        log(classification_report(r['y_true'], r['y_pred'], labels=labels,\n"
        "                                  target_names=class_names, zero_division=0))"
    ))

    # learning curve grid
    cells.append(md(
        "## 9. Learning Curve — cek Overfit / Underfit (tiap teknik)\n\n"
        "Grid: baris = model, kolom = teknik. Mode resample (classifier tanpa "
        "balanced); SVC pakai `probability=False` agar cepat."
    ))
    cells.append(code(
        "from sklearn.model_selection import learning_curve, StratifiedKFold\n\n"
        "LC_TRAIN_SIZES = np.linspace(0.2, 1.0, 5)\n"
        "LC_CV = 5; LC_SCORING = 'balanced_accuracy'; LC_STATE = 42\n"
        "groups_arr = np.asarray(groups)\n"
        "techs = list(RESAMPLERS.keys())\n\n"
        "lc = {}\n"
        "for name in MODELS:\n"
        "    for tech in techs:\n"
        "        log(f'  > LC: {name} | {tech} ...')\n"
        "        try:\n"
        "            pipe = make_resample_pipeline(name, BEST_PARAMS[name], RESAMPLERS[tech],\n"
        "                                          n_optimal, xgb_device=XGB_DEVICE, cat_task_type=CAT_TASK_TYPE)\n"
        "            ts, tr, va = learning_curve(\n"
        "                pipe, X, y, groups=groups_arr, train_sizes=LC_TRAIN_SIZES,\n"
        "                cv=StratifiedKFold(LC_CV, shuffle=True, random_state=LC_STATE),\n"
        "                scoring=LC_SCORING, n_jobs=1, shuffle=True, random_state=LC_STATE)\n"
        "            lc[(name, tech)] = (ts, tr.mean(1), tr.std(1), va.mean(1), va.std(1))\n"
        "        except Exception as e:\n"
        "            log(f'    FAILED [{name}|{tech}] -> {type(e).__name__}: {e}')\n"
        "log(f'LC selesai: {len(lc)} kurva.')"
    ))
    cells.append(code(
        "nrows = len(MODELS); ncols = len(techs)\n"
        "fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), constrained_layout=True)\n"
        "axes = np.atleast_2d(axes)\n"
        "for i, name in enumerate(MODELS):\n"
        "    for j, tech in enumerate(techs):\n"
        "        ax = axes[i, j]\n"
        "        if (name, tech) not in lc:\n"
        "            ax.axis('off'); continue\n"
        "        ts, trm, trs, vam, vas = lc[(name, tech)]\n"
        "        ax.plot(ts, trm, 'o-', color='steelblue', lw=2, label='Train')\n"
        "        ax.fill_between(ts, trm - trs, trm + trs, alpha=0.2, color='steelblue')\n"
        "        ax.plot(ts, vam, 's-', color='darkorange', lw=2, label='Val')\n"
        "        ax.fill_between(ts, vam - vas, vam + vas, alpha=0.2, color='darkorange')\n"
        "        gap = trm[-1] - vam[-1]\n"
        "        v = ('OVERFIT' if gap > 0.15 else 'UNDERFIT' if vam[-1] < 0.5\n"
        "             else 'Well-fit' if gap < 0.05 and vam[-1] > 0.7 else 'Mild gap')\n"
        "        ax.set_title(f'{name} | {tech}\\ngap={gap:+.3f} | {v}', fontsize=9, fontweight='bold')\n"
        "        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.4); ax.legend(fontsize=7)\n"
        "fig.suptitle(f'Learning Curve — {STRATEGY} (model tanpa balanced) | {LC_SCORING}',\n"
        "             fontsize=13, fontweight='bold')\n"
        "fig.savefig(os.path.join(OUT_DIR, f'{STRATEGY}_learning_curve.png'), dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 10. Kesimpulan\n\n"
        "- **Tabel best-per-model (sel 6)**: strategi mana menang di tiap model.\n"
        "- Bandingkan SMOTEENN vs SMOTETomek vs adjusted di bar chart & confusion "
        "matrix (recall kelas **B**).\n"
        "- Learning curve (sel 9): cek apakah hybrid bikin overfit (gap besar).\n"
        "- Semua dari `StratifiedKFold(Sampling_ID)` + resampling khusus fold "
        "training → bebas bocor.\n\n"
        "File hasil di `OUT_DIR`."
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for key, scn in SCENARIOS.items():
        nb = build(key, scn)
        out = os.path.join(here, f"{key}.ipynb")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("wrote", out)
    for key, scn in MULTI_SCENARIOS.items():
        nb = build_multi(scn)
        out = os.path.join(here, f"{key}.ipynb")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("wrote", out)
