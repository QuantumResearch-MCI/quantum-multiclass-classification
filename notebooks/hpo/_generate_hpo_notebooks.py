'''
Generator for the split HPO notebooks.

Produces one notebook per model (and per circuit/model for quantum) under
``notebooks/hpo/``, plus a single master ``results.ipynb`` that aggregates every
saved best-param artifact.

This script ONLY creates new files; it does not touch the original notebooks
(3.1_hpo_classical_all.ipynb, 3.2_hpo_classical_all_dl.ipynb,
3.6_hpo_quantum_pqk.ipynb). Re-running it regenerates the split notebooks.

The HPO pipeline / scoring is kept identical to the original notebooks. On top
of that, every notebook now has:
  * resumable checkpoints (skip already-evaluated configs),
  * a best-param fit diagnostic plot (train vs. validation -> overfit /
    underfit / generalized),
  * a saved JSON best-result artifact consumed by results.ipynb.

Run from anywhere:
    python notebooks/hpo/_generate_hpo_notebooks.py
'''

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../notebooks/hpo
CLASSICAL_DIR = HERE / "classical"
DL_DIR = HERE / "dl"
QUANTUM_DIR = HERE / "quantum"


# --------------------------------------------------------------------------- #
# Low-level notebook helpers
# --------------------------------------------------------------------------- #
def _src(text):
    """Notebook 'source' wants a list of lines (each keeping its newline)."""
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def write_notebook(path: Path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  wrote {path.relative_to(HERE.parent.parent)}")


def fill(template, **kw):
    """Token replacement using %%TOKEN%% so f-string braces survive."""
    out = template
    for k, v in kw.items():
        out = out.replace(f"%%{k}%%", str(v))
    return out


# --------------------------------------------------------------------------- #
# Shared code blocks (templates)
# --------------------------------------------------------------------------- #
INSTALL_CELL = r'''
# !python -m pip install numpy pandas \
#     qiskit-aer qiskit-algorithms qiskit-machine-learning qiskit-ibm-runtime \
#     pylatexenc ucimlrepo \
#     xgboost catboost seaborn libsvm-official \
#     jinja2 scikit-optimize tensorflow imbalanced-learn
'''

ENV_CELL = r'''
import sys
import os

cwd = os.path.abspath(os.getcwd())
project_root = cwd.split("quantum-multiclass-classification")[0] + "quantum-multiclass-classification"
print(f"Project root: {project_root}")

sys.path.append(os.path.abspath(project_root))

from utils.prepare_data import prepare_data
'''

PRELUDE_CELL = r'''
import pandas as pd
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

import sys
import importlib

def reload_package(package_name):
    modules_to_reload = [name for name in sys.modules if name.startswith(package_name)]
    for name in sorted(modules_to_reload, reverse=True):
        importlib.reload(sys.modules[name])

reload_package("model")
reload_package("utils")
'''

DATA_CELL = r'''
dataset_path = os.path.join(project_root, "dataset", "Dataset_TehHijau.csv")
feature_cols = [
        "MQ3", "TGS822", "TGS2602", "MQ5", "MQ138", "TGS2620",
        "TGS813", "TGS2600", "TGS2611", "TGS2603", "Humidity", "Celsius",
    ]
target_cols = "Kategori"

import pandas as pd
data = pd.read_csv(dataset_path)
data.head(10)
'''

XY_CELL = r'''
# ── ⚙️ Skenario Eksperimen ──────────────────────────────────
# DROP_CLASSES   : label kelas (nilai asli kolom target) yang dibuang sebelum split.
#                  Contoh ["B"] -> buang kelas B. [] = pakai semua kelas.
# CUSTOM_TARGETS : target jumlah per-kelas saat resample (dihitung per fold, train fold only):
#                  {} = tanpa resampling. nilai = int (jumlah absolut) ATAU nama kelas lain
#                  (samakan jumlahnya), mis. {"B": "C", "E": "C"}. Kelas tak disebut = dibiarkan.
#                  "auto-over"  = full oversampling (semua ke mayoritas);
#                  "auto-under" = full undersampling (semua ke minoritas).
# OVER_METHOD    : teknik oversampling -> 'smote' | 'borderline' | 'smoteenn' | 'smotetomek'
# UNDER_METHOD   : teknik undersampling -> 'random' (target eksak) | 'tomek' (cleaning)
DROP_CLASSES   = []
CUSTOM_TARGETS = {}
OVER_METHOD    = "smoteenn"
UNDER_METHOD   = "random"

if DROP_CLASSES:
    _before = len(data)
    data = data[~data[target_cols].isin(DROP_CLASSES)].reset_index(drop=True)
    print(f"\U0001f5d1️  Drop kelas {DROP_CLASSES}: {_before} -> {len(data)} baris")
print("Distribusi kelas:", dict(data[target_cols].value_counts().sort_index()))

def make_resampler():
    """Resampler kustom per-kelas (CUSTOM_TARGETS) via utils.imbalance_eval.
    None bila CUSTOM_TARGETS kosong (tanpa resampling). Target dihitung per fold (no leak)."""
    if not CUSTOM_TARGETS:
        return None
    from utils.imbalance_eval import make_custom_resampler
    return make_custom_resampler(
        CUSTOM_TARGETS, list(label_encoder.classes_),
        over_method=OVER_METHOD, under_method=UNDER_METHOD)

def resampler_steps():
    """Step pipeline imblearn untuk resampler terpilih ([] bila 'none')."""
    r = make_resampler()
    return [("resampler", r)] if r is not None else []

X = data[feature_cols]
y = data[target_cols]
print(X.shape, y.shape)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
'''

PCA_CELL = r'''
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_pca_variance(X, n_components, threshold=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_optimal = np.argmax(cumvar >= threshold) + 1

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), cumvar, marker='o', color='blue')
    plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, n_components + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f'Optimal number of components to retain {threshold*100:.0f}% variance: {n_optimal}')
    return n_optimal

n_optimal = plot_pca_variance(X, n_components=X.shape[1], threshold=%%PCA_THRESHOLD%%)
'''

# Paths + logger + checkpoint + save_best.  CATEGORY controls sub-folders so
# every slug stays collision-free across the whole hpo/ tree.
INFRA_CELL = r'''
import time, json, pickle
import numpy as np
from pathlib import Path
from datetime import datetime

CATEGORY = "%%CATEGORY%%"
# Bump when the HPO logic changes in a way that invalidates old checkpoints.
# v2: dropped class_weight/sample_weight balancing + overfit-aware selection.
# v3: added SMOTEENN resampling inside the CV pipeline (train-fold only).
# v4: scoring switched to penalized validation log-loss (minimize), not composite.
# v5: overfit penalty now uses TWO signals — loss_gap + macro-F1 gap.
CKPT_VERSION = 5
HPO_RESULTS = Path(project_root) / "notebooks" / "hpo" / "results"
CSV_DIR  = HPO_RESULTS / "csv" / CATEGORY
LOG_DIR  = HPO_RESULTS / "logs" / CATEGORY
CKPT_DIR = HPO_RESULTS / "checkpoints" / CATEGORY
BEST_DIR = HPO_RESULTS / "best"
FIG_DIR  = HPO_RESULTS / "figures" / CATEGORY
for _d in (CSV_DIR, LOG_DIR, CKPT_DIR, BEST_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_current_log_path = None

def setup_logger(name):
    """Call at the start of each section; name is the section slug."""
    global _current_log_path
    _current_log_path = LOG_DIR / f"{name}.log"
    _current_log_path.write_text("")          # fresh log every run
    log(f"📝 Log: {_current_log_path}")
    log(f"🕒 Started: {datetime.now().isoformat(timespec='seconds')}")

def log(msg=""):
    print(msg)
    if _current_log_path is not None:
        with open(_current_log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

def _fresh_ckpt():
    # best_score = +inf because selection minimizes penalized log-loss.
    return {"version": CKPT_VERSION, "done_configs": {}, "results": [],
            "best_result": None, "best_score": np.inf}

def load_checkpoint(name):
    """Resume a section; returns an empty skeleton if no (compatible) checkpoint."""
    ckpt_path = CKPT_DIR / f"{name}.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CKPT_VERSION:
            log(f"♻️  Ignoring stale checkpoint (v{data.get('version')} != v{CKPT_VERSION}); starting fresh: {ckpt_path}")
            return _fresh_ckpt(), ckpt_path
        log(f"♻️  Resumed checkpoint: {ckpt_path} ({len(data['done_configs'])} configs done)")
        return data, ckpt_path
    return _fresh_ckpt(), ckpt_path

def save_checkpoint(ckpt_path, data):
    """Atomic save (write .tmp then replace)."""
    data.setdefault("version", CKPT_VERSION)
    tmp = ckpt_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(ckpt_path)

def save_best(name, model_label, best_result, extra=None):
    """Persist the section's best result so results.ipynb can aggregate it."""
    payload = {
        "name": name,
        "model": model_label,
        "category": CATEGORY,
        "tag": best_result.get("tag"),
        "loss": best_result.get("loss"),
        "selection_score": best_result.get("selection_score"),
        "train_loss_cv": best_result.get("train_loss_cv"),
        "train_f1macro_cv": best_result.get("train_f1macro_cv"),
        "loss_gap": best_result.get("loss_gap"),
        "f1_gap": best_result.get("f1_gap"),
        "acc": best_result.get("acc"),
        "prec": best_result.get("prec"),
        "rec": best_result.get("rec"),
        "f1": best_result.get("f1"),
        "f1_macro": best_result.get("f1_macro"),
        "roc": best_result.get("roc"),
        "pra": best_result.get("pra"),
        "execution_time": best_result.get("execution_time"),
        "params": best_result.get("params", {}),
    }
    if extra:
        payload.update(extra)
    with open(BEST_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log(f"💾 Best saved: {BEST_DIR / (name + '.json')}")
'''

# Fit diagnostic for sklearn-style estimators (classical + quantum).
DIAG_SKLEARN_CELL = r'''
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

def plot_fit_diagnostic(build_estimator, params, name, title,
                        normalize_proba=False, fit_kwargs_fn=None,
                        overfit_gap=0.05, low_thresh=0.80):
    """Refit best params per fold, compare TRAIN vs VAL to flag over/underfit."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _fkw = fit_kwargs_fn or (lambda y_tr: {})

    def _scores(model, Xs, ys):
        yp = model.predict(Xs)
        pr = model.predict_proba(Xs)
        if normalize_proba:
            pr = pr / pr.sum(axis=1, keepdims=True)
        a = accuracy_score(ys, yp)
        ll = log_loss(ys, pr, labels=np.unique(y))
        return a, ll

    tr_acc, va_acc, tr_loss, va_loss = [], [], [], []
    for tr, va in skf.split(X, y):
        m = build_estimator(params)
        m.fit(X.iloc[tr], y[tr], **_fkw(y[tr]))
        a1, c1 = _scores(m, X.iloc[tr], y[tr])
        a2, c2 = _scores(m, X.iloc[va], y[va])
        tr_acc.append(a1); va_acc.append(a2); tr_loss.append(c1); va_loss.append(c2)

    tr_acc_m, va_acc_m = float(np.mean(tr_acc)), float(np.mean(va_acc))
    gap = tr_acc_m - va_acc_m
    # Overfit (train >> val) is checked first so a big gap is never masked as
    # "Underfit" just because both means sit below low_thresh.
    if gap > overfit_gap:
        verdict = "Overfit"
    elif va_acc_m < low_thresh:
        verdict = "Underfit"
    else:
        verdict = "Generalized"

    folds = np.arange(1, 6)
    w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(folds - w/2, tr_acc, w, label='Train', color='#4C72B0')
    ax[0].bar(folds + w/2, va_acc, w, label='Validation', color='#DD8452')
    ax[0].axhline(tr_acc_m, color='#4C72B0', ls='--', lw=1)
    ax[0].axhline(va_acc_m, color='#DD8452', ls='--', lw=1)
    ax[0].set_title('Accuracy'); ax[0].set_xlabel('Fold'); ax[0].set_ylim(0, 1.05)
    ax[0].set_xticks(folds); ax[0].legend(); ax[0].grid(axis='y', alpha=0.3)

    ax[1].bar(folds - w/2, tr_loss, w, label='Train', color='#4C72B0')
    ax[1].bar(folds + w/2, va_loss, w, label='Validation', color='#DD8452')
    ax[1].set_title('Log-loss (lower = better)')
    ax[1].set_xlabel('Fold')
    ax[1].set_xticks(folds); ax[1].legend(); ax[1].grid(axis='y', alpha=0.3)

    fig.suptitle(f"{title} — best-param fit: {verdict}  "
                 f"(train acc={tr_acc_m:.3f}, val acc={va_acc_m:.3f}, gap={gap:+.3f})",
                 fontsize=12)
    fig.tight_layout()
    fig_path = FIG_DIR / f"{name}_fit.png"
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.show()

    log(f"📈 Fit diagnostic [{title}] → {verdict} "
        f"(train acc={tr_acc_m:.4f} | val acc={va_acc_m:.4f} | gap={gap:+.4f})")
    log(f"🖼️  Figure: {fig_path}")
    return {"fit_verdict": verdict, "train_acc": tr_acc_m,
            "val_acc": va_acc_m, "fit_gap": float(gap)}
'''

# Generic sklearn HPO loop (classical + quantum).  Relies on globals:
#   build_estimator, NORMALIZE_PROBA, search_space, X, y, n_optimal,
#   log, setup_logger, load_checkpoint, save_checkpoint, np, time.
SK_LOOP_CELL = r'''
setup_logger("%%NAME%%")
from sklearn.model_selection import StratifiedKFold
from itertools import product
import time
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    precision_score, recall_score, log_loss,
)

state = 42
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)

classes = np.unique(y)

# Primary scoring = mean validation LOG-LOSS (lower is better). Overfit is checked
# with TWO signals, each penalized only beyond its own tolerance:
#   loss_gap = val_loss - train_loss        -> probability calibration
#   f1_gap   = train_macroF1 - val_macroF1  -> per-class balance (catches minority overfit)
#   penalty  = W_LOSS*max(0, loss_gap - TOL_LOSS) + W_F1*max(0, f1_gap - TOL_F1)
#   selection_score = val_loss + penalty    -> best config = the MINIMUM
# Train metrics are measured on the ORIGINAL (non-resampled) train fold.
TOL_LOSS = 0.05   # "free" train-val log-loss gap (no penalty below this)
TOL_F1   = 0.05   # "free" train-val macro-F1 gap
W_LOSS   = 1.0    # weight for the log-loss overfit signal
W_F1     = 1.0    # weight for the macro-F1 overfit signal

param_keys = list(search_space.keys())
param_vals = list(search_space.values())
search_space_sizes = {k: len(v) for k, v in search_space.items()}
total_configs = int(np.prod([len(v) for v in param_vals]))
total_fits = total_configs * skf.get_n_splits()
space_str = " × ".join([f"{search_space_sizes[k]} {k}" for k in search_space_sizes])

log(f"🔬 Search space: {space_str} =  {total_configs} configs × {skf.get_n_splits()} folds = {total_fits} fits")
log("   Scoring criterion: mean validation log-loss (lower better); overfit penalty on loss_gap + macroF1_gap")
log(f"   Overfit guard: +{W_LOSS}×max(0, loss_gap - {TOL_LOSS}) +{W_F1}×max(0, f1_gap - {TOL_F1})")

ckpt, ckpt_path = load_checkpoint("%%NAME%%")
done_configs = ckpt["done_configs"]
results      = ckpt["results"]
best_result  = ckpt["best_result"]
best_score   = ckpt["best_score"]

for i, comb in enumerate(product(*param_vals)):
    params = dict(zip(param_keys, comb))
    tag = " | ".join(f"{k}={v}" for k, v in params.items())

    if tag in done_configs:
        log(f"\n  ⏭️  [{i+1}/{total_configs}] {tag}  (skipped, already done)")
        continue

    log(f"\n  ▶ [{i+1}/{total_configs}] {tag}")
    _iter_t0 = time.perf_counter()

    accs, f1s, f1ms, rocs, pras, precs, recs, lls = [], [], [], [], [], [], [], []
    tr_lls, tr_f1ms = [], []
    y_val_all, y_pred_all = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_estimator(params)
        model.fit(X_train, y_train, **fit_kwargs_fn(y_train))

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        if NORMALIZE_PROBA:
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        # Train log-loss + macro-F1 (original train fold) for the overfit checks
        tr_prob = model.predict_proba(X_train)
        if NORMALIZE_PROBA:
            tr_prob = tr_prob / tr_prob.sum(axis=1, keepdims=True)
        tr_lls.append(log_loss(y_train, tr_prob, labels=classes))
        # use model.predict (not argmax of proba) so train f1 matches how val f1 is computed
        tr_f1ms.append(f1_score(y_train, model.predict(X_train), average='macro'))

        y_val_all.extend(y_val)
        y_pred_all.extend(y_pred)

        acc  = accuracy_score(y_val, y_pred)
        f1   = f1_score(y_val, y_pred, average='weighted')
        f1m  = f1_score(y_val, y_pred, average='macro')
        roc  = roc_auc_score(y_val, y_prob, average='weighted', multi_class='ovr')
        pra  = average_precision_score(y_val, y_prob, average='weighted')
        prec = precision_score(y_val, y_pred, average='weighted')
        rec  = recall_score(y_val, y_pred, average='weighted')
        ll   = log_loss(y_val, y_prob, labels=classes)

        accs.append(acc); f1s.append(f1); f1ms.append(f1m); rocs.append(roc)
        pras.append(pra); precs.append(prec); recs.append(rec); lls.append(ll)

        log(f"    F{fold} → Loss={ll:.4f} | Acc={acc:.4f} | F1w={f1:.4f} | F1m={f1m:.4f} | AUROC={roc:.4f} | PRAUC={pra:.4f}")

        results.append({
            "tag": tag, **params, "fold": fold,
            "accuracy": acc, "precision": prec, "recall": rec,
            "F1": f1, "f1_macro": f1m, "auroc": roc, "prauc": pra,
            "loss": ll, "train_loss": tr_lls[-1],
        })

    acc_mean,  acc_std  = np.mean(accs),  np.std(accs)
    prec_mean, prec_std = np.mean(precs), np.std(precs)
    rec_mean,  rec_std  = np.mean(recs),  np.std(recs)
    f1_mean,   f1_std   = np.mean(f1s),   np.std(f1s)
    f1m_mean,  f1m_std  = np.mean(f1ms),  np.std(f1ms)
    roc_mean,  roc_std  = np.mean(rocs),  np.std(rocs)
    pra_mean,  pra_std  = np.mean(pras),  np.std(pras)
    loss_mean, loss_std = np.mean(lls),   np.std(lls)
    mcc = matthews_corrcoef(y_val_all, y_pred_all)

    # Overfit-aware selection: MINIMIZE val log-loss, penalize TWO overfit signals:
    #   loss_gap = val_loss - train_loss ; f1_gap = train_macroF1 - val_macroF1
    tr_loss_mean = float(np.mean(tr_lls))
    tr_f1m_mean  = float(np.mean(tr_f1ms))
    loss_gap = loss_mean - tr_loss_mean
    f1_gap   = tr_f1m_mean - f1m_mean
    penalty  = W_LOSS * max(0.0, loss_gap - TOL_LOSS) + W_F1 * max(0.0, f1_gap - TOL_F1)
    selection_score = loss_mean + penalty
    overfit_flag = " ⚠️ OVERFIT" if (loss_gap > TOL_LOSS or f1_gap > TOL_F1) else ""

    log(
        f"  ✅  Loss:{loss_mean:.4f}±{loss_std:.4f} | "
        f"Acc:{acc_mean:.4f}±{acc_std:.4f} | "
        f"F1w:{f1_mean:.4f}±{f1_std:.4f} | "
        f"F1macro:{f1m_mean:.4f}±{f1m_std:.4f} | "
        f"AUROC:{roc_mean:.4f}±{roc_std:.4f} | "
        f"PRAUC:{pra_mean:.4f}±{pra_std:.4f} | "
        f"MCC:{mcc:.4f}"
    )
    log(
        f"      ↳ LossGap:{loss_gap:+.4f} (tr {tr_loss_mean:.4f}) | "
        f"F1macroGap:{f1_gap:+.4f} (tr {tr_f1m_mean:.4f}) | "
        f"Penalty:{penalty:.4f} | Selection(loss):{selection_score:.4f}{overfit_flag}"
    )

    iter_time = time.perf_counter() - _iter_t0
    if best_score is None or selection_score < best_score:
        best_score = selection_score
        best_result = {
            'tag': tag,
            'loss': f"{loss_mean:.4f}±{loss_std:.4f}",
            'selection_score': selection_score,
            'train_loss_cv': f"{tr_loss_mean:.4f}",
            'train_f1macro_cv': f"{tr_f1m_mean:.4f}",
            'loss_gap': f"{loss_gap:+.4f}",
            'f1_gap': f"{f1_gap:+.4f}",
            'acc':  f"{acc_mean:.4f}±{acc_std:.4f}",
            'f1':   f"{f1_mean:.4f}±{f1_std:.4f}",
            'f1_macro': f"{f1m_mean:.4f}±{f1m_std:.4f}",
            'roc':  f"{roc_mean:.4f}±{roc_std:.4f}",
            'pra':  f"{pra_mean:.4f}±{pra_std:.4f}",
            'prec': f"{prec_mean:.4f}±{prec_std:.4f}",
            'rec':  f"{rec_mean:.4f}±{rec_std:.4f}",
            'params': params, 'execution_time': iter_time,
        }

    done_configs[tag] = {"loss": loss_mean, "selection_score": selection_score,
                         "loss_gap": loss_gap, "f1_gap": f1_gap, "params": params}
    save_checkpoint(ckpt_path, {
        "version": CKPT_VERSION,
        "done_configs": done_configs, "results": results,
        "best_result": best_result, "best_score": best_score,
    })

log(f"\n🏆 Best config : {best_result['tag']}  (min penalized log-loss)")
log(
    f"   Log-loss    : {best_result['loss']} "
    f"(Acc={best_result['acc']} | F1w={best_result['f1']} | F1macro={best_result['f1_macro']} | "
    f"AUROC={best_result['roc']} | PRAUC={best_result['pra']})"
)
log(
    f"   Selection   : {best_result['selection_score']:.4f} "
    f"(LossGap={best_result['loss_gap']} | F1macroGap={best_result['f1_gap']})"
)
'''

SAVE_CSV_CELL = r'''
import pandas as pd
df = pd.DataFrame(results)
filename = CSV_DIR / "%%NAME%%_hpo.csv"
df.to_csv(filename, index=False)
log(f"✅ Saved: {filename}")
'''

DIAG_CALL_CELL = r'''
# 📈 Best-param fit diagnostic (overfit / underfit / generalized)
fit_info = plot_fit_diagnostic(
    build_estimator, best_result['params'],
    name="%%NAME%%", title="%%TITLE%%", normalize_proba=NORMALIZE_PROBA,
    fit_kwargs_fn=fit_kwargs_fn,
)
'''

RECORD_BEST_CELL = r'''
best_result['model'] = "%%MODEL_LABEL%%"
save_best("%%NAME%%", "%%MODEL_LABEL%%", best_result, extra={%%EXTRA%%, **fit_info})
log(f"✅ [%%MODEL_LABEL%%] recorded | Exec. time: {best_result['execution_time']:.1f}s")
'''


# --------------------------------------------------------------------------- #
# Section builder for sklearn-style models (classical + quantum)
# --------------------------------------------------------------------------- #
def sk_section(header_md, search_space_src, build_estimator_src, name, title,
               model_label, normalize_proba, extra_dict_src,
               fit_kwargs_src="fit_kwargs_fn = lambda y_tr: {}"):
    """A full per-model(-variant) section: header, config, loop, csv, diag, record.

    `fit_kwargs_src` defines `fit_kwargs_fn(y_train)` -> dict of extra fit kwargs
    passed to estimator.fit (defaults to none; kept as a hook for per-fold needs).
    """
    config = (
        "# Search Space + estimator builder\n"
        "import numpy as np\n\n"
        f"NORMALIZE_PROBA = {normalize_proba}\n\n"
        f"{search_space_src.strip()}\n\n"
        f"{build_estimator_src.strip()}\n\n"
        f"{fit_kwargs_src.strip()}\n"
    )
    return [
        md(header_md),
        code(config),
        code(fill(SK_LOOP_CELL, NAME=name)),
        code(fill(SAVE_CSV_CELL, NAME=name)),
        code(fill(DIAG_CALL_CELL, NAME=name, TITLE=title)),
        code(fill(RECORD_BEST_CELL, NAME=name, MODEL_LABEL=model_label,
                  EXTRA=extra_dict_src)),
    ]


def common_header(title_md, category, pca_threshold, extra_setup_cells=None):
    cells = [
        md(title_md),
        code(INSTALL_CELL),
        code(ENV_CELL),
        code(PRELUDE_CELL),
        code(DATA_CELL),
        code(XY_CELL),
        code(fill(PCA_CELL, PCA_THRESHOLD=pca_threshold)),
        code(fill(INFRA_CELL, CATEGORY=category)),
        code(DIAG_SKLEARN_CELL),
    ]
    if extra_setup_cells:
        cells.extend(extra_setup_cells)
    return cells


# --------------------------------------------------------------------------- #
# CLASSICAL notebooks
# --------------------------------------------------------------------------- #
def gen_classical():
    print("Classical:")

    # ---- SVC (linear / poly / rbf / sigmoid) -----------------------------
    # Fixed (anti-overfit): ovr, gamma='scale'. Class imbalance is handled by a
    # SMOTEENN step inside the CV pipeline (train-fold only), so no
    # class_weight='balanced' here.
    # Tuned: C (regularization strength) + kernel-shape params only. degree capped
    # at 2-3 (degree=5 was overfit-prone); gamma no longer tuned.
    svc_specs = [
        ("linear", "## Linear",
         "search_space = {\n  'C': [0.01, 0.1, 1],\n}"),
        ("poly", "## Poly",
         "search_space = {\n  'C': [0.01, 0.1, 1],\n  'degree': [2, 3],\n  'coef0': [0.0, 1.0],\n}"),
        ("rbf", "## RBF",
         "search_space = {\n  'C': [0.01, 0.1, 1],\n}"),
        ("sigmoid", "## Sigmoid",
         "search_space = {\n  'C': [0.01, 0.1, 1],\n  'coef0': [0.0, 0.5, 1.0],\n}"),
    ]
    svc_disp = {"linear": "Linear", "poly": "Poly", "rbf": "RBF", "sigmoid": "Sigmoid"}
    for kern, header, ss in svc_specs:        # one notebook per kernel
        disp = svc_disp[kern]
        gamma_arg = "" if kern == "linear" else "gamma='scale', "
        build = (
            "from imblearn.pipeline import Pipeline\n"
            "from imblearn.combine import SMOTEENN\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.svm import SVC\n\n"
            "def build_estimator(params):\n"
            "    return Pipeline([\n"
            "        ('scaler', StandardScaler()),\n"
            "        ('pca', PCA(n_components=n_optimal)),\n"
            "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
            f"        ('svc', SVC(kernel='{kern}', {gamma_arg}\n"
            "                    probability=True, random_state=42,\n"
            "                    decision_function_shape='ovr', **params)),\n"
            "    ])"
        )
        cells = common_header(f"# SVC {disp} — Hyperparameter Optimization", "classical", 0.92)
        cells += sk_section(header, ss, build, f"svc_{kern}", f"SVC {disp}",
                            f"SVC {disp}", False, "'variant': 'svc'")
        write_notebook(CLASSICAL_DIR / f"svc_{kern}.ipynb", cells)

    # ---- XGBoost (gbtree / dart) — one notebook per booster --------------
    # Fixed: subsample/colsample_bytree=0.8 (row/col subsampling). Class imbalance
    # is handled by a SMOTEENN step inside the CV pipeline (train-fold only), so no
    # balanced sample_weight here. Tuned now includes regularizers
    # min_child_weight & reg_lambda; max_depth lowered 4-10 -> 3-6.
    # Trimmed to ~hundreds: every regularizer kept but at 2 values; subsample &
    # colsample_bytree fixed at 0.8 (builder). scale_pos_weight dropped (binary-
    # only; imbalance handled upstream by resampling). gbtree=2^7=128 combos.
    gbtree_ss = ("search_space = {\n  'n_estimators': [200, 500],\n  'learning_rate': [0.01, 0.1],\n"
                 "  'max_depth': [3, 6],\n  'min_child_weight': [1, 10],\n"
                 "  'reg_alpha': [0, 1],\n  'reg_lambda': [1, 10],\n  'gamma': [0, 1],\n}")
    # dart adds tree-dropout (rate_drop, skip_drop) -> 2^9=512 combos.
    dart_ss = ("search_space = {\n  'n_estimators': [200, 500],\n  'learning_rate': [0.01, 0.1],\n"
               "  'max_depth': [3, 6],\n  'min_child_weight': [1, 10],\n"
               "  'reg_alpha': [0, 1],\n  'reg_lambda': [1, 10],\n  'gamma': [0, 1],\n"
               "  'rate_drop': [0.1, 0.3],\n  'skip_drop': [0.0, 0.5],\n}")
    xgb_disp = {"gbtree": "GBTree", "dart": "Dart"}
    for booster, header, ss in [("gbtree", "## GBTree", gbtree_ss), ("dart", "## Dart", dart_ss)]:
        disp = xgb_disp[booster]
        build = (
            "from imblearn.pipeline import Pipeline\n"
            "from imblearn.combine import SMOTEENN\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.decomposition import PCA\n"
            "from xgboost import XGBClassifier\n\n"
            "def build_estimator(params):\n"
            "    return Pipeline([\n"
            "        ('scaler', StandardScaler()),\n"
            "        ('pca', PCA(n_components=n_optimal)),\n"
            "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
            f"        ('xgb', XGBClassifier(booster='{booster}', objective='multi:softprob',\n"
            "                              subsample=0.8, colsample_bytree=0.8,\n"
            "                              random_state=42, device='cuda', **params)),\n"
            "    ])"
        )
        cells = common_header(f"# XGBoost {disp} — Hyperparameter Optimization", "classical", 0.92)
        cells += sk_section(header, ss, build, f"xgb_{booster}",
                            f"XGBoost {disp}", f"XGBoost {disp}",
                            False, "'variant': 'xgboost'")
        write_notebook(CLASSICAL_DIR / f"xgboost_{booster}.ipynb", cells)

    # ---- CatBoost --------------------------------------------------------
    # Fixed: MultiClassOneVsAll (OvR). Class imbalance is handled by a SMOTEENN
    # step inside the CV pipeline (train-fold only), so no auto_class_weights here.
    # Anti-overfit: depth capped 4->6, stronger l2_leaf_reg (3-7).
    # Trimmed (2^6=64). bagging_temperature needs bootstrap_type='Bayesian' (fixed).
    cells = common_header("# CatBoost — Hyperparameter Optimization", "classical", 0.92)
    cat_ss = ("search_space = {\n  'iterations': [200, 500],\n  'depth': [4, 6],\n"
              "  'learning_rate': [0.03, 0.1],\n  'l2_leaf_reg': [3, 10],\n"
              "  'bagging_temperature': [0, 1],\n  'random_strength': [1, 5],\n}")
    build = (
        "from imblearn.pipeline import Pipeline\n"
        "from imblearn.combine import SMOTEENN\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.decomposition import PCA\n"
        "from catboost import CatBoostClassifier\n\n"
        "def build_estimator(params):\n"
        "    return Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('pca', PCA(n_components=n_optimal)),\n"
        "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
        "        ('cat', CatBoostClassifier(loss_function='MultiClassOneVsAll',\n"
        "                                   eval_metric='Accuracy', verbose=0,\n"
        "                                   bootstrap_type='Bayesian',\n"
        "                                   random_seed=42, **params)),\n"
        "    ])"
    )
    cells += sk_section("## CatBoost", cat_ss, build, "catboost", "CatBoost",
                        "CatBoost", True, "'variant': 'catboost'")
    write_notebook(CLASSICAL_DIR / "catboost.ipynb", cells)


# --------------------------------------------------------------------------- #
# DEEP-LEARNING notebooks
# --------------------------------------------------------------------------- #
TF_SETUP_CELL = r'''
# TensorFlow / Keras setup: deterministic seed + GPU memory growth
import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

for _g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

n_classes = int(len(np.unique(y)))
print(f'TF {tf.__version__} | GPUs detected: {len(tf.config.list_physical_devices("GPU"))}')
print(f'n_classes = {n_classes}')
'''

# DL loop + diagnostic share one per-fold prep routine (prep_fold).
# Deep learning uses 5-fold StratifiedKFold CV (same protocol as the sklearn models).
DL_HELPERS_CELL = r'''
# Shared per-fold scale + PCA (+ optional Conv reshape) for the DL models.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN

def make_resampler():
    """Resampler kustom per-kelas (CUSTOM_TARGETS) via utils.imbalance_eval.
    None bila CUSTOM_TARGETS kosong (tanpa resampling). Target dihitung per fold (no leak)."""
    if not CUSTOM_TARGETS:
        return None
    from utils.imbalance_eval import make_custom_resampler
    return make_custom_resampler(
        CUSTOM_TARGETS, list(label_encoder.classes_),
        over_method=OVER_METHOD, under_method=UNDER_METHOD)

def prep_fold(train_idx, val_idx, conv=False, resampler=None):
    """Scale + PCA per fold. If `resampler` given, resample the TRAIN fold only
    (in PCA space, before the Conv reshape) -> no leakage into validation.
    Returns the resampled train set (for .fit) AND the original train set
    (for the overfit-gap measurement), plus the validation set."""
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    pca = PCA(n_components=n_optimal)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p   = pca.transform(X_val_s)
    # Original (non-resampled) train kept for the overfit-gap measurement.
    X_train_eval, y_train_eval = X_train_p, y_train
    if resampler is not None:
        X_train_fit, y_train_fit = resampler.fit_resample(X_train_p, y_train)
    else:
        X_train_fit, y_train_fit = X_train_p, y_train
    if conv:
        X_train_fit  = X_train_fit[..., np.newaxis]
        X_train_eval = X_train_eval[..., np.newaxis]
        X_val_p      = X_val_p[..., np.newaxis]
    return X_train_fit, y_train_fit, X_train_eval, y_train_eval, X_val_p, y_val

def plot_fit_diagnostic_dl(build_fn, params, name, title, conv=False,
                           overfit_gap=0.05, low_thresh=0.80):
    """Refit best DL params per fold; compare TRAIN vs VAL."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, log_loss
    import tensorflow as tf
    from tensorflow import keras

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _scores(model, Xs, ys):
        pr = model.predict(Xs, verbose=0)
        yp = np.argmax(pr, axis=1)
        a = accuracy_score(ys, yp)
        ll = log_loss(ys, pr, labels=np.unique(y))
        return a, ll

    tr_acc, va_acc, tr_loss, va_loss = [], [], [], []
    for tr, va in skf.split(X, y):
        Xtr, ytr, Xtr_eval, ytr_eval, Xva, yva = prep_fold(
            tr, va, conv=conv, resampler=make_resampler())
        keras.backend.clear_session()
        model = build_fn(Xtr.shape[1], n_classes, params)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                           restore_best_weights=True, verbose=0)
        model.fit(Xtr, ytr, validation_data=(Xva, yva),
                  epochs=params['epochs'], batch_size=params['batch_size'],
                  verbose=0, callbacks=[es])
        # train scored on ORIGINAL (non-resampled) fold so the gap is honest
        a1, c1 = _scores(model, Xtr_eval, ytr_eval)
        a2, c2 = _scores(model, Xva, yva)
        tr_acc.append(a1); va_acc.append(a2); tr_loss.append(c1); va_loss.append(c2)

    tr_acc_m, va_acc_m = float(np.mean(tr_acc)), float(np.mean(va_acc))
    gap = tr_acc_m - va_acc_m
    if gap > overfit_gap:
        verdict = "Overfit"
    elif va_acc_m < low_thresh:
        verdict = "Underfit"
    else:
        verdict = "Generalized"

    folds = np.arange(1, 6); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(folds - w/2, tr_acc, w, label='Train', color='#4C72B0')
    ax[0].bar(folds + w/2, va_acc, w, label='Validation', color='#DD8452')
    ax[0].axhline(tr_acc_m, color='#4C72B0', ls='--', lw=1)
    ax[0].axhline(va_acc_m, color='#DD8452', ls='--', lw=1)
    ax[0].set_title('Accuracy'); ax[0].set_xlabel('Fold'); ax[0].set_ylim(0, 1.05)
    ax[0].set_xticks(folds); ax[0].legend(); ax[0].grid(axis='y', alpha=0.3)
    ax[1].bar(folds - w/2, tr_loss, w, label='Train', color='#4C72B0')
    ax[1].bar(folds + w/2, va_loss, w, label='Validation', color='#DD8452')
    ax[1].set_title('Log-loss (lower = better)')
    ax[1].set_xlabel('Fold')
    ax[1].set_xticks(folds); ax[1].legend(); ax[1].grid(axis='y', alpha=0.3)
    fig.suptitle(f"{title} — best-param fit: {verdict}  "
                 f"(train acc={tr_acc_m:.3f}, val acc={va_acc_m:.3f}, gap={gap:+.3f})",
                 fontsize=12)
    fig.tight_layout()
    fig_path = FIG_DIR / f"{name}_fit.png"
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.show()
    log(f"📈 Fit diagnostic [{title}] → {verdict} "
        f"(train acc={tr_acc_m:.4f} | val acc={va_acc_m:.4f} | gap={gap:+.4f})")
    return {"fit_verdict": verdict, "train_acc": tr_acc_m,
            "val_acc": va_acc_m, "fit_gap": float(gap)}
'''

DL_LOOP_CELL = r'''
setup_logger("%%NAME%%")
from sklearn.model_selection import StratifiedKFold
from itertools import product
import time
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    precision_score, recall_score, log_loss,
)
import tensorflow as tf
from tensorflow import keras

state = 42
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)

classes = np.unique(y)

# 5-fold StratifiedKFold CV. Primary scoring = mean validation LOG-LOSS (lower is
# better). Overfit is checked with TWO signals, each penalized only beyond its tolerance:
#   loss_gap = val_loss - train_loss        -> probability calibration
#   f1_gap   = train_macroF1 - val_macroF1  -> per-class balance (catches minority overfit)
#   penalty  = W_LOSS*max(0, loss_gap - TOL_LOSS) + W_F1*max(0, f1_gap - TOL_F1)
#   selection_score = mean val_loss + penalty    -> best config = the MINIMUM
# Train metrics are measured on the ORIGINAL (non-resampled) train fold.
TOL_LOSS = 0.05   # "free" train-val log-loss gap (no penalty below this)
TOL_F1   = 0.05   # "free" train-val macro-F1 gap
W_LOSS   = 1.0    # weight for the log-loss overfit signal
W_F1     = 1.0    # weight for the macro-F1 overfit signal

param_keys = list(search_space.keys())
param_vals = list(search_space.values())
search_space_sizes = {k: len(v) for k, v in search_space.items()}
total_configs = int(np.prod([len(v) for v in param_vals]))
total_fits = total_configs * skf.get_n_splits()
space_str = " × ".join([f"{search_space_sizes[k]} {k}" for k in search_space_sizes])

log(f"🔬 Search space: {space_str} =  {total_configs} configs × {skf.get_n_splits()} folds = {total_fits} fits")
log("   Scoring criterion: mean validation log-loss (lower better); overfit penalty on loss_gap + macroF1_gap")
log(f"   Overfit guard: +{W_LOSS}×max(0, loss_gap - {TOL_LOSS}) +{W_F1}×max(0, f1_gap - {TOL_F1})")

ckpt, ckpt_path = load_checkpoint("%%NAME%%")
# Reverted to 5-fold CV; drop any single hold-out checkpoint (dl_protocol mismatch)
# so we never resume stale hold-out results, without touching classical/quantum.
DL_PROTOCOL = "kfold-v2"
if ckpt.get("dl_protocol") != DL_PROTOCOL:
    if ckpt["done_configs"]:
        log("♻️  Ignoring checkpoint from a different DL protocol; starting fresh.")
    ckpt = _fresh_ckpt()
    ckpt["dl_protocol"] = DL_PROTOCOL
done_configs = ckpt["done_configs"]
results      = ckpt["results"]
best_result  = ckpt["best_result"]
best_score   = ckpt["best_score"]

for i, comb in enumerate(product(*param_vals)):
    params = dict(zip(param_keys, comb))
    tag = " | ".join(f"{k}={v}" for k, v in params.items())

    if tag in done_configs:
        log(f"\n  ⏭️  [{i+1}/{total_configs}] {tag}  (skipped, already done)")
        continue

    log(f"\n  ▶ [{i+1}/{total_configs}] {tag}")
    _iter_t0 = time.perf_counter()

    accs, f1s, f1ms, rocs, pras, precs, recs, lls = [], [], [], [], [], [], [], []
    tr_lls, tr_f1ms = [], []
    y_val_all, y_pred_all = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # SMOTEENN resamples the TRAIN fold only (no leakage into validation).
        X_train_fit, y_train_fit, X_train_eval, y_train_eval, X_val_p, y_val = prep_fold(
            train_idx, val_idx, conv=CONV, resampler=make_resampler())

        keras.backend.clear_session()
        model = build_model(X_train_fit.shape[1], n_classes, params)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                           restore_best_weights=True, verbose=0)
        model.fit(X_train_fit, y_train_fit, validation_data=(X_val_p, y_val),
                  epochs=params['epochs'], batch_size=params['batch_size'],
                  verbose=0, callbacks=[es])

        y_prob = model.predict(X_val_p, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        # Train log-loss + macro-F1 on ORIGINAL (non-resampled) fold for the overfit checks
        tr_prob_eval = model.predict(X_train_eval, verbose=0)
        tr_lls.append(log_loss(y_train_eval, tr_prob_eval, labels=classes))
        tr_f1ms.append(f1_score(y_train_eval, np.argmax(tr_prob_eval, axis=1), average='macro'))

        y_val_all.extend(y_val)
        y_pred_all.extend(y_pred)

        acc  = accuracy_score(y_val, y_pred)
        f1   = f1_score(y_val, y_pred, average='weighted')
        f1m  = f1_score(y_val, y_pred, average='macro')
        roc  = roc_auc_score(y_val, y_prob, average='weighted', multi_class='ovr')
        pra  = average_precision_score(y_val, y_prob, average='weighted')
        prec = precision_score(y_val, y_pred, average='weighted')
        rec  = recall_score(y_val, y_pred, average='weighted')
        ll   = log_loss(y_val, y_prob, labels=classes)

        accs.append(acc); f1s.append(f1); f1ms.append(f1m); rocs.append(roc)
        pras.append(pra); precs.append(prec); recs.append(rec); lls.append(ll)

        log(f"    F{fold} → Loss={ll:.4f} | Acc={acc:.4f} | F1w={f1:.4f} | F1m={f1m:.4f} | AUROC={roc:.4f} | PRAUC={pra:.4f}")

        results.append({
            "tag": tag, **params, "fold": fold,
            "accuracy": acc, "precision": prec, "recall": rec,
            "F1": f1, "f1_macro": f1m, "auroc": roc, "prauc": pra,
            "loss": ll, "train_loss": tr_lls[-1],
        })

    acc_mean,  acc_std  = np.mean(accs),  np.std(accs)
    prec_mean, prec_std = np.mean(precs), np.std(precs)
    rec_mean,  rec_std  = np.mean(recs),  np.std(recs)
    f1_mean,   f1_std   = np.mean(f1s),   np.std(f1s)
    f1m_mean,  f1m_std  = np.mean(f1ms),  np.std(f1ms)
    roc_mean,  roc_std  = np.mean(rocs),  np.std(rocs)
    pra_mean,  pra_std  = np.mean(pras),  np.std(pras)
    loss_mean, loss_std = np.mean(lls),   np.std(lls)
    mcc = matthews_corrcoef(y_val_all, y_pred_all)

    # Overfit-aware selection: MINIMIZE mean val log-loss, penalize TWO overfit signals.
    tr_loss_mean = float(np.mean(tr_lls))
    tr_f1m_mean  = float(np.mean(tr_f1ms))
    loss_gap = loss_mean - tr_loss_mean
    f1_gap   = tr_f1m_mean - f1m_mean
    penalty  = W_LOSS * max(0.0, loss_gap - TOL_LOSS) + W_F1 * max(0.0, f1_gap - TOL_F1)
    selection_score = loss_mean + penalty
    overfit_flag = " ⚠️ OVERFIT" if (loss_gap > TOL_LOSS or f1_gap > TOL_F1) else ""

    log(
        f"  ✅  Loss:{loss_mean:.4f}±{loss_std:.4f} | "
        f"Acc:{acc_mean:.4f}±{acc_std:.4f} | "
        f"F1w:{f1_mean:.4f}±{f1_std:.4f} | "
        f"F1macro:{f1m_mean:.4f}±{f1m_std:.4f} | "
        f"AUROC:{roc_mean:.4f}±{roc_std:.4f} | "
        f"PRAUC:{pra_mean:.4f}±{pra_std:.4f} | "
        f"MCC:{mcc:.4f}"
    )
    log(
        f"      ↳ LossGap:{loss_gap:+.4f} (tr {tr_loss_mean:.4f}) | "
        f"F1macroGap:{f1_gap:+.4f} (tr {tr_f1m_mean:.4f}) | "
        f"Penalty:{penalty:.4f} | Selection(loss):{selection_score:.4f}{overfit_flag}"
    )

    iter_time = time.perf_counter() - _iter_t0
    if best_score is None or selection_score < best_score:
        best_score = selection_score
        best_result = {
            'tag': tag,
            'loss': f"{loss_mean:.4f}±{loss_std:.4f}",
            'selection_score': selection_score,
            'train_loss_cv': f"{tr_loss_mean:.4f}",
            'train_f1macro_cv': f"{tr_f1m_mean:.4f}",
            'loss_gap': f"{loss_gap:+.4f}",
            'f1_gap': f"{f1_gap:+.4f}",
            'acc':  f"{acc_mean:.4f}±{acc_std:.4f}",
            'f1':   f"{f1_mean:.4f}±{f1_std:.4f}",
            'f1_macro': f"{f1m_mean:.4f}±{f1m_std:.4f}",
            'roc':  f"{roc_mean:.4f}±{roc_std:.4f}",
            'pra':  f"{pra_mean:.4f}±{pra_std:.4f}",
            'prec': f"{prec_mean:.4f}±{prec_std:.4f}",
            'rec':  f"{rec_mean:.4f}±{rec_std:.4f}",
            'params': params, 'execution_time': iter_time,
        }

    done_configs[tag] = {"loss": loss_mean, "selection_score": selection_score,
                         "loss_gap": loss_gap, "f1_gap": f1_gap, "params": params}
    save_checkpoint(ckpt_path, {
        "version": CKPT_VERSION, "dl_protocol": DL_PROTOCOL,
        "done_configs": done_configs, "results": results,
        "best_result": best_result, "best_score": best_score,
    })

log(f"\n🏆 Best config : {best_result['tag']}  (min penalized log-loss)")
log(
    f"   Log-loss    : {best_result['loss']} "
    f"(Acc={best_result['acc']} | F1w={best_result['f1']} | F1macro={best_result['f1_macro']} | "
    f"AUROC={best_result['roc']} | PRAUC={best_result['pra']})"
)
log(
    f"   Selection   : {best_result['selection_score']:.4f} "
    f"(LossGap={best_result['loss_gap']} | F1macroGap={best_result['f1_gap']})"
)
'''

DL_DIAG_CALL = r'''
# 📈 Best-param fit diagnostic (overfit / underfit / generalized) — 5-fold CV
fit_info = plot_fit_diagnostic_dl(
    build_model, best_result['params'],
    name="%%NAME%%", title="%%TITLE%%", conv=CONV,
)
'''


def gen_dl():
    print("Deep learning:")
    extra = [code(TF_SETUP_CELL), code(DL_HELPERS_CELL)]

    # ---- MLP -------------------------------------------------------------
    cells = common_header("# MLP — Hyperparameter Optimization", "dl", 0.92, extra)
    mlp_cfg = (
        "# Search Space + model builder — MLP\n"
        "import numpy as np\n"
        "from tensorflow import keras\n"
        "from tensorflow.keras import layers, regularizers\n\n"
        "CONV = False\n"
        "L2 = 1e-4   # weight decay (anti-overfit). Imbalance handled by SMOTEENN on the train split\n\n"
        "search_space = {\n"
        "  'hidden_units' : [(64, 32), (128, 64, 32)],\n"
        "  'dropout'      : [0.2, 0.4],\n"
        "  'learning_rate': [1e-3, 1e-4],\n"
        "  'batch_size'   : [32, 64],\n"
        "  'epochs'       : [100],\n"
        "}\n\n"
        "def build_model(input_dim, n_classes, params):\n"
        "    keras.utils.set_random_seed(42)\n"
        "    inputs = keras.Input(shape=(input_dim,))\n"
        "    x = inputs\n"
        "    for h in params['hidden_units']:\n"
        "        x = layers.Dense(h, activation='relu', kernel_initializer='he_normal',\n"
        "                         kernel_regularizer=regularizers.l2(L2))(x)\n"
        "        x = layers.BatchNormalization()(x)\n"
        "        x = layers.Dropout(params['dropout'])(x)\n"
        "    outputs = layers.Dense(n_classes, activation='softmax')(x)\n"
        "    model = keras.Model(inputs, outputs)\n"
        "    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),\n"
        "                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
        "    return model"
    )
    cells += [
        md("## MLP"),
        code(mlp_cfg),
        code(fill(DL_LOOP_CELL, NAME="mlp")),
        code(fill(SAVE_CSV_CELL, NAME="mlp")),
        code(fill(DL_DIAG_CALL, NAME="mlp", TITLE="MLP")),
        code(fill(RECORD_BEST_CELL, NAME="mlp", MODEL_LABEL="MLP", EXTRA="'variant': 'mlp'")),
    ]
    write_notebook(DL_DIR / "mlp.ipynb", cells)

    # ---- 1D CNN ----------------------------------------------------------
    cells = common_header("# 1D CNN — Hyperparameter Optimization", "dl", 0.92, extra)
    cnn_cfg = (
        "# Search Space + model builder — 1D CNN\n"
        "import numpy as np\n"
        "from tensorflow import keras\n"
        "from tensorflow.keras import layers, regularizers\n\n"
        "CONV = True\n"
        "L2 = 1e-4   # weight decay (anti-overfit). Imbalance handled by SMOTEENN on the train split\n\n"
        "# kernel_size dikunci ke 3 (pilihan paling umum) supaya budget pencarian CNN\n"
        "# setara dengan MLP: 16 konfigurasi vs 16 konfigurasi.\n"
        "search_space = {\n"
        "  'filters'      : [(32, 64), (64, 128)],\n"
        "  'kernel_size'  : [3],\n"
        "  'dropout'      : [0.2, 0.4],\n"
        "  'learning_rate': [1e-3, 1e-4],\n"
        "  'batch_size'   : [32, 64],\n"
        "  'epochs'       : [100],\n"
        "}\n\n"
        "def build_model(input_len, n_classes, params):\n"
        "    keras.utils.set_random_seed(42)\n"
        "    inputs = keras.Input(shape=(input_len, 1))\n"
        "    x = inputs\n"
        "    for f in params['filters']:\n"
        "        x = layers.Conv1D(f, kernel_size=params['kernel_size'], padding='same',\n"
        "                          activation='relu', kernel_initializer='he_normal',\n"
        "                          kernel_regularizer=regularizers.l2(L2))(x)\n"
        "        x = layers.BatchNormalization()(x)\n"
        "        if x.shape[1] is not None and x.shape[1] > 1:\n"
        "            x = layers.MaxPooling1D(pool_size=2)(x)\n"
        "        x = layers.Dropout(params['dropout'])(x)\n"
        "    x = layers.GlobalAveragePooling1D()(x)\n"
        "    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2))(x)\n"
        "    x = layers.Dropout(params['dropout'])(x)\n"
        "    outputs = layers.Dense(n_classes, activation='softmax')(x)\n"
        "    model = keras.Model(inputs, outputs)\n"
        "    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),\n"
        "                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
        "    return model"
    )
    cells += [
        md("## 1D CNN"),
        code(cnn_cfg),
        code(fill(DL_LOOP_CELL, NAME="cnn1d")),
        code(fill(SAVE_CSV_CELL, NAME="cnn1d")),
        code(fill(DL_DIAG_CALL, NAME="cnn1d", TITLE="1D CNN")),
        code(fill(RECORD_BEST_CELL, NAME="cnn1d", MODEL_LABEL="1D CNN", EXTRA="'variant': 'cnn1d'")),
    ]
    write_notebook(DL_DIR / "cnn1d.ipynb", cells)


# --------------------------------------------------------------------------- #
# QUANTUM notebooks
# --------------------------------------------------------------------------- #
# family folder -> {file base: actual QKERNEL circuit registered in estimator.py}
QUANTUM_FAMILIES = {
    "original": {
        "full": "full", "linear": "linear", "circular": "circular",
        "pauli_x": "pauli_x", "pauli_y": "pauli_y", "pauli_z": "pauli_z",
    },
    "quadratic": {"full": "full_quadratic", "linear": "linear_quadratic", "circular": "circular_quadratic"},
    "cosine": {"full": "full_cosine", "linear": "linear_cosine", "circular": "circular_cosine"},
    "selisih": {"full": "full_selisih", "linear": "linear_selisih", "circular": "circular_selisih"},
    "polynomial": {"full": "full_polynomial", "linear": "linear_polynomial", "circular": "circular_polynomial"},
    "polynomial4": {"full": "full_polynomial4", "linear": "linear_polynomial4", "circular": "circular_polynomial4"},
    "x-base": {"full": "x_full", "linear": "x_linear", "circular": "x_circular"},
    "y-base": {"full": "y_full", "linear": "y_linear", "circular": "y_circular"},
    "topology": {"star": "star"},
}

Q_MODE_CELL = r'''
# Quantum global configuration
mode = 'pqk'
QKERNEL = '%%QKERNEL%%'
print(f"mode={mode} | QKERNEL={QKERNEL}")
'''

# Quantum grids are aggressively trimmed: every fit recomputes a costly
# Projected Quantum Kernel, so we tune only a few core regularizers + lambda_.
# n_measurements is commented out (the PQK kernel does not use shots).
QSVC_SS = ("search_space = {\n  'C': [0.01, 0.1, 1],\n  # quantum params\n"
           "  'lambda_': [0.01, 0.1, 1, 5],\n  # 'n_measurements': [256, 1024],   # PQK ignores n_measurements\n}")
QXGB_GB_SS = ("search_space = {\n  'learning_rate': [0.05, 0.1],\n  'max_depth': [3, 6],\n"
              "  'reg_lambda': [1, 10],\n  # quantum params\n  'lambda_': [0.1, 1],\n"
              "  # 'n_measurements': [256, 1024],   # PQK ignores n_measurements\n}")
QXGB_DART_SS = ("search_space = {\n  'learning_rate': [0.05, 0.1],\n  'max_depth': [3, 6],\n"
                "  'reg_lambda': [1, 10],\n  'rate_drop': [0.1, 0.3],\n  # quantum params\n  'lambda_': [0.1, 1],\n"
                "  # 'n_measurements': [256, 1024],   # PQK ignores n_measurements\n}")
QCAT_SS = ("search_space = {\n  'depth': [4, 6],\n  'learning_rate': [0.03, 0.1],\n"
           "  'l2_leaf_reg': [3, 10],\n  # quantum params\n  'lambda_': [0.1, 1],\n"
           "  # 'n_measurements': [256, 1024],   # PQK ignores n_measurements\n}")

QSVC_BUILD = (
    "from imblearn.pipeline import Pipeline\n"
    "from imblearn.combine import SMOTEENN\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.decomposition import PCA\n"
    "from model.quantum.qsvc import QSVCWrapper\n\n"
    "def build_estimator(params):\n"
    "    return Pipeline([\n"
    "        ('scaler', StandardScaler()),\n"
    "        ('pca', PCA(n_components=n_optimal)),\n"
    "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
    "        ('svc', QSVCWrapper(kernel=QKERNEL, mode=mode, n_qubits=n_optimal,\n"
    "                            n_features=n_optimal, random_state=42,\n"
    "                            decision_function_shape='ovr', **params)),\n"
    "    ])"
)


def _qxgb_build(booster):
    # Hardcoded regularizers (disjoint from the tuned grid: learning_rate,
    # max_depth, reg_lambda [+ rate_drop for dart]). Class imbalance is handled by
    # a SMOTEENN step inside the CV pipeline (train-fold only), so no balanced
    # sample_weight is injected here.
    fixed = ("n_estimators=300, min_child_weight=5,\n"
             "                      subsample=0.8, colsample_bytree=0.8,\n"
             "                      reg_alpha=0.0, gamma=0.0,")
    if booster == "dart":
        fixed += "\n                      skip_drop=0.5,"
    return (
        "from imblearn.pipeline import Pipeline\n"
        "from imblearn.combine import SMOTEENN\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.decomposition import PCA\n"
        "from model.quantum.qxgb import QXGB\n\n"
        "def build_estimator(params):\n"
        "    return Pipeline([\n"
        "        ('scaler', StandardScaler()),\n"
        "        ('pca', PCA(n_components=n_optimal)),\n"
        "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
        f"        ('qxgb', QXGB(kernel=QKERNEL, mode=mode, n_qubits=n_optimal,\n"
        f"                      n_features=n_optimal, booster='{booster}',\n"
        f"                      {fixed}\n"
        "                      random_state=42, **params)),\n"
        "    ])"
    )


QCAT_BUILD = (
    "from imblearn.pipeline import Pipeline\n"
    "from imblearn.combine import SMOTEENN\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.decomposition import PCA\n"
    "from model.quantum.qcat import QCAT\n\n"
    "def build_estimator(params):\n"
    "    return Pipeline([\n"
    "        ('scaler', StandardScaler()),\n"
    "        ('pca', PCA(n_components=n_optimal)),\n"
    "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
    "        ('qcat', QCAT(kernel=QKERNEL, mode=mode, n_qubits=n_optimal,\n"
    "                      n_features=n_optimal, iterations=300,\n"
    "                      bootstrap_type='Bayesian', bagging_temperature=1,\n"
    "                      random_strength=1,\n"
    "                      random_seed=42, **params)),\n"
    "    ])"
)


# VQC is VARIATIONAL (not a kernel): it trains a parameterized ansatz via a
# classical optimizer and is VERY slow (every optimizer iteration runs the
# circuit over all train samples). The grid is kept tiny on purpose. The feature
# map is borrowed from the kernel estimator so QKERNEL + lambda_ stay consistent
# with QSVC dkk; mode/n_measurements are accepted for call-uniformity but unused
# (the sampler is statevector). ansatz/entanglement/reps drive the trainable
# circuit; optimizer/maxiter drive the classical training loop.
VQC_SS = ("search_space = {\n  'reps': [1, 2],\n  'optimizer': ['cobyla'],\n  'maxiter': [100],\n"
          "  # quantum params\n  'ansatz': ['real_amplitudes'],\n  'entanglement': ['full'],\n"
          "  'lambda_': [0.1, 1],\n  # 'n_measurements': [256, 1024],   # VQC uses a statevector sampler\n}")

VQC_BUILD = (
    "from imblearn.pipeline import Pipeline\n"
    "from imblearn.combine import SMOTEENN\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.decomposition import PCA\n"
    "from model.quantum.vqc import VQCWrapper\n\n"
    "def build_estimator(params):\n"
    "    return Pipeline([\n"
    "        ('scaler', StandardScaler()),\n"
    "        ('pca', PCA(n_components=n_optimal)),\n"
    "        *resampler_steps(),  # resampling (train-fold only) sesuai CUSTOM_TARGETS\n"
    "        ('vqc', VQCWrapper(kernel=QKERNEL, mode=mode, n_qubits=n_optimal,\n"
    "                           n_features=n_optimal, random_state=42, **params)),\n"
    "    ])"
)


def gen_quantum():
    print("Quantum:")
    total = 0
    for family, mapping in QUANTUM_FAMILIES.items():
        for base, circuit in mapping.items():
            extra_common = f"'family': '{family}', 'circuit': '{circuit}'"

            # -- QSVC notebook --
            cells = common_header(f"# QSVC `{circuit}` — Hyperparameter Optimization (PQK)",
                                  "quantum", 0.95, [code(fill(Q_MODE_CELL, QKERNEL=circuit))])
            name = f"quantum_{family}_{base}_qsvc"
            cells += sk_section("## QSVC", QSVC_SS, QSVC_BUILD, name,
                                f"QSVC ({circuit})", f"QSVC ({circuit})", False, extra_common)
            write_notebook(QUANTUM_DIR / family / f"{base}_qsvc.ipynb", cells)
            total += 1

            # -- QXGB GBTree notebook --
            cells = common_header(f"# QXGB GBTree `{circuit}` — Hyperparameter Optimization (PQK)",
                                  "quantum", 0.95, [code(fill(Q_MODE_CELL, QKERNEL=circuit))])
            cells += sk_section("## QXGB — GBTree", QXGB_GB_SS, _qxgb_build("gbtree"),
                                f"quantum_{family}_{base}_qxgb_gbtree",
                                f"QXGB GBTree ({circuit})", f"QXGB GBTree ({circuit})",
                                False, extra_common)
            write_notebook(QUANTUM_DIR / family / f"{base}_qxgb_gbtree.ipynb", cells)
            total += 1

            # -- QXGB Dart notebook --
            cells = common_header(f"# QXGB Dart `{circuit}` — Hyperparameter Optimization (PQK)",
                                  "quantum", 0.95, [code(fill(Q_MODE_CELL, QKERNEL=circuit))])
            cells += sk_section("## QXGB — Dart", QXGB_DART_SS, _qxgb_build("dart"),
                                f"quantum_{family}_{base}_qxgb_dart",
                                f"QXGB Dart ({circuit})", f"QXGB Dart ({circuit})",
                                False, extra_common)
            write_notebook(QUANTUM_DIR / family / f"{base}_qxgb_dart.ipynb", cells)
            total += 1

            # -- QCat notebook --
            cells = common_header(f"# QCat `{circuit}` — Hyperparameter Optimization (PQK)",
                                  "quantum", 0.95, [code(fill(Q_MODE_CELL, QKERNEL=circuit))])
            name = f"quantum_{family}_{base}_qcat"
            cells += sk_section("## QCat", QCAT_SS, QCAT_BUILD, name,
                                f"QCat ({circuit})", f"QCat ({circuit})", True, extra_common)
            write_notebook(QUANTUM_DIR / family / f"{base}_qcat.ipynb", cells)
            total += 1

            # -- VQC notebook (variational, not a kernel) --
            cells = common_header(f"# VQC `{circuit}` — Hyperparameter Optimization (variational)",
                                  "quantum", 0.95, [code(fill(Q_MODE_CELL, QKERNEL=circuit))])
            name = f"quantum_{family}_{base}_vqc"
            cells += sk_section("## VQC", VQC_SS, VQC_BUILD, name,
                                f"VQC ({circuit})", f"VQC ({circuit})", False, extra_common)
            write_notebook(QUANTUM_DIR / family / f"{base}_vqc.ipynb", cells)
            total += 1
    print(f"  ({total} quantum notebooks)")


# --------------------------------------------------------------------------- #
# MASTER results notebook
# --------------------------------------------------------------------------- #
RESULTS_LOAD_CELL = r'''
import json
import pandas as pd
from pathlib import Path

BEST_DIR = Path(project_root) / "notebooks" / "hpo" / "results" / "best"
files = sorted(BEST_DIR.glob("*.json"))
print(f"Found {len(files)} best-param artifacts in {BEST_DIR}")

rows = []
for fp in files:
    with open(fp, encoding="utf-8") as f:
        r = json.load(f)
    params_str = ' | '.join(f'{k}={v}' for k, v in r.get('params', {}).items())
    rows.append({
        'Category'       : r.get('category'),
        'Family'         : r.get('family', ''),
        'Circuit/Kernel' : r.get('circuit', ''),
        'Model'          : r.get('model'),
        'Accuracy'       : r.get('acc'),
        'Precision'      : r.get('prec'),
        'Recall'         : r.get('rec'),
        'F1-Score'       : r.get('f1'),
        'F1-Macro'       : r.get('f1_macro'),
        'ROC-AUC'        : r.get('roc'),
        'PR-AUC'         : r.get('pra'),
        'Log-Loss'       : r.get('loss'),
        'Selection'      : r.get('selection_score'),
        'Loss Gap'       : r.get('loss_gap'),
        'F1 Gap'         : r.get('f1_gap'),
        'Fit'            : r.get('fit_verdict', ''),
        'Train Acc'      : r.get('train_acc'),
        'Val Acc'        : r.get('val_acc'),
        'Fit Gap'        : r.get('fit_gap'),
        'Exec. Time (s)' : round(r['execution_time'], 2) if r.get('execution_time') is not None else None,
        'Best Params'    : params_str,
        'Artifact'       : fp.name,
    })

all_df = pd.DataFrame(rows)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.4f}'.format)
print(f"Total evaluations: {len(all_df)}")
all_df.head()
'''

RESULTS_TABLE_CELL = r'''
# Full ranking by selection score (penalized val log-loss, lower = better)
if len(all_df):
    ranked = all_df.sort_values('Selection', ascending=True).reset_index(drop=True)
    ranked.index += 1
    out = Path(project_root) / "notebooks" / "hpo" / "results" / "eval_all_final_hpo.csv"
    ranked.to_csv(out, index=True)
    print(f"✅ Saved combined ranking: {out}")
    display(ranked)
else:
    print("No artifacts yet — run the model notebooks first.")
'''

RESULTS_PERCAT_CELL = r'''
# Per-category leaderboards
for cat, g in all_df.groupby('Category'):
    print(f"\n===== {cat.upper()} ({len(g)} models) =====")
    gg = g.sort_values('Selection', ascending=True).reset_index(drop=True)
    gg.index += 1
    display(gg[['Family', 'Circuit/Kernel', 'Model', 'Accuracy', 'F1-Macro',
                'Log-Loss', 'Selection', 'Fit', 'Best Params']])
'''

RESULTS_PLOT_CELL = r'''
import matplotlib.pyplot as plt
import numpy as np

if len(all_df):
    # 1) Selection score per model (top 30 = lowest penalized log-loss), by category
    plot_df = all_df.dropna(subset=['Selection']).sort_values('Selection', ascending=False)
    if len(plot_df) > 30:
        plot_df = plot_df.tail(30)   # keep the 30 best (lowest selection)
    cats = plot_df['Category'].astype('category')
    palette = {c: clr for c, clr in zip(cats.cat.categories,
               plt.cm.tab10.colors)}
    colors = [palette[c] for c in plot_df['Category']]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(plot_df))))
    ax.barh(plot_df['Model'], plot_df['Selection'].astype(float), color=colors)
    ax.set_xlabel('Selection = penalized val log-loss (lower = better)')
    ax.set_title('Best selection score per model (top 30, lower = better)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in palette]
    ax.legend(handles, list(palette.keys()), title='Category')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(project_root) / "notebooks" / "hpo" / "results" / "summary_selection.png",
                dpi=120, bbox_inches='tight')
    plt.show()

    # 2) Fit verdict counts
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = all_df['Fit'].value_counts()
    ax.bar(vc.index, vc.values, color=['#55A868', '#C44E52', '#8172B2', '#999999'][:len(vc)])
    ax.set_title('Best-param fit diagnosis across all models')
    ax.set_ylabel('Count')
    for i, v in enumerate(vc.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(Path(project_root) / "notebooks" / "hpo" / "results" / "summary_fit.png",
                dpi=120, bbox_inches='tight')
    plt.show()

    # 3) Train vs Val accuracy scatter (generalization view)
    sub = all_df.dropna(subset=['Train Acc', 'Val Acc'])
    if len(sub):
        fig, ax = plt.subplots(figsize=(6, 6))
        for c in sub['Category'].unique():
            s = sub[sub['Category'] == c]
            ax.scatter(s['Train Acc'].astype(float), s['Val Acc'].astype(float),
                       label=c, alpha=0.8)
        lim = [0.0, 1.02]
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('Train Accuracy'); ax.set_ylabel('Validation Accuracy')
        ax.set_title('Train vs Validation (points below diagonal = overfit)')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(project_root) / "notebooks" / "hpo" / "results" / "summary_trainval.png",
                    dpi=120, bbox_inches='tight')
        plt.show()
'''


def gen_results():
    print("Results:")
    cells = [
        md("# 📊 HPO Results — Master Summary\n\n"
           "Aggregates every saved best-param artifact under "
           "`notebooks/hpo/results/best/` (classical, deep learning, and quantum). "
           "Run the model notebooks first; this notebook only reads their outputs, "
           "so it can be re-run any time to refresh the leaderboard."),
        code(ENV_CELL),
        code(RESULTS_LOAD_CELL),
        md("## Combined leaderboard"),
        code(RESULTS_TABLE_CELL),
        md("## Per-category leaderboards"),
        code(RESULTS_PERCAT_CELL),
        md("## Summary plots"),
        code(RESULTS_PLOT_CELL),
    ]
    write_notebook(HERE / "results.ipynb", cells)


if __name__ == "__main__":
    gen_classical()
    gen_dl()
    gen_quantum()
    gen_results()
    print("\nDone.")
