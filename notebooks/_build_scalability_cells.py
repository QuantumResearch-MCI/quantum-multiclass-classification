"""Modify notebook 4 to hardcode 8 scalability cells (2 kernels x 4 sizes).

Each cell is self-contained: load dataset, set kernel/search_space,
run main loop (identical to comparison_qsvc_* cells above), record best.
"""
import json, uuid
from pathlib import Path

NB = Path(__file__).with_name("4. qsvc_iqp_vs_cust_iqp.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def newid():
    return uuid.uuid4().hex[:8]


def md(src):
    return {
        "cell_type": "markdown",
        "id": newid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "id": newid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CONFIG_CELL = '''# Konfigurasi pengujian skalabilitas
scalability_datasets = {
    500:   os.path.join(project_root, "dataset", "Dataset_TehHijau500.csv"),
    1000:  os.path.join(project_root, "dataset", "Dataset_TehHijau1k.csv"),
    5000:  os.path.join(project_root, "dataset", "Dataset_TehHijau5k.csv"),
    10000: os.path.join(project_root, "dataset", "Dataset_TehHijau.csv"),
}

# Dict untuk menampung best result per (dataset_size, kernel)
scalability_results = {}
'''


def cell_for(SIZE, KERNEL):
    return f'''# ════ QSVC ({KERNEL}) @ size = {SIZE} ════
mode = 'fsk'
QKERNEL = '{KERNEL}'
DATASET_SIZE = {SIZE}

# Load dataset
df_local = pd.read_csv(scalability_datasets[DATASET_SIZE])
X = df_local[feature_cols]
y = LabelEncoder().fit_transform(df_local[target_cols])

# n_optimal untuk dataset ini (95% varians)
_pca_probe = PCA(n_components=X.shape[1]).fit(StandardScaler().fit_transform(X))
n_optimal = int(np.argmax(np.cumsum(_pca_probe.explained_variance_ratio_) >= 0.95) + 1)
print(f"📦 size={{DATASET_SIZE}} | X={{X.shape}} | n_optimal={{n_optimal}}")

# Search Space Configuration
search_space = {{
  'C': [1],

  # quantum params
  'lambda_': [0.3],
  # 'n_measurements': [1024],
}}

param_keys = search_space.keys()
param_vals = search_space.values()

setup_logger(f"scalability_qsvc_{{QKERNEL}}_{{DATASET_SIZE}}")
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model.quantum.qsvc import QSVCWrapper
from sklearn.decomposition import PCA
from itertools import product
from sklearn.metrics import (
  accuracy_score, f1_score, roc_auc_score,
  average_precision_score, matthews_corrcoef,
  precision_score, recall_score
)

state = 42
n_splits = 5
n_iter = 10

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)

search_space_sizes = {{k: len(v) for k, v in search_space.items()}}
total_configs = np.prod([len(v) for v in param_vals])
total_fits = total_configs * skf.get_n_splits()
space_str = " × ".join([f"{{search_space_sizes[k]}} {{k}}" for k in search_space_sizes])

log(f"🔬 Dataset={{DATASET_SIZE}} | Kernel={{QKERNEL}} | Search space: {{space_str}} =  {{total_configs}} configs × {{skf.get_n_splits()}} folds = {{total_fits}} fits")
log("   Scoring criterion: (AUROC + PRAUC + Accuracy) / 3")

ckpt, ckpt_path = load_checkpoint(f"scalability_qsvc_{{QKERNEL}}_{{DATASET_SIZE}}")
done_configs = ckpt["done_configs"]
results = ckpt["results"]
best_result = ckpt["best_result"]
best_score = ckpt["best_score"]

for i, comb in enumerate(product(*param_vals)):
  params = dict(zip(param_keys, comb))
  tag = " | ".join(f"{{k}}={{v}}" for k, v in params.items())

  if tag in done_configs:
    log(f"\\n  ⏭️  [{{i+1}}/{{total_configs}}] {{tag}}  (skipped, already done)")
    continue

  log(f"\\n  ▶ [{{i+1}}/{{total_configs}}] {{tag}}")
  _iter_t0 = time.perf_counter()

  accs, f1s, rocs, pras, precs, recs= [], [], [], [], [], []
  y_val_all, y_pred_all = [], []
  for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Pipeline([
      ('scaler', StandardScaler()),
      ('pca', PCA(n_components=n_optimal)),
      ('svc', QSVCWrapper(
        kernel=QKERNEL,
        mode=mode,
        n_qubits=n_optimal,
        n_features=n_optimal,

        random_state=42,
        decision_function_shape='ovr',
        **params
      ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    y_val_all.extend(y_val)
    y_pred_all.extend(y_pred)


    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    roc = roc_auc_score(y_val, y_prob, average='weighted', multi_class='ovr')
    pra = average_precision_score(y_val, y_prob, average='weighted')
    prec = precision_score(y_val, y_pred, average='weighted')
    rec = recall_score(y_val, y_pred, average='weighted')

    accs.append(acc)
    f1s.append(f1)
    rocs.append(roc)
    pras.append(pra)
    precs.append(prec)
    recs.append(rec)

    log(f"    F{{fold}} → Acc={{acc:.4f}} | Prec={{prec:.4f}} | Rec={{rec:.4f}} | F1={{f1:.4f}} | AUROC={{roc:.4f}} | PRAUC={{pra:.4f}}")

    results.append({{
      "tag": tag,
      **params,
      "fold": fold,
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "F1": f1,
      "auroc": roc,
      "prauc": pra,
    }})

  acc_mean, acc_std = np.mean(accs), np.std(accs)
  prec_mean, prec_std = np.mean(precs), np.std(precs)
  rec_mean, rec_std = np.mean(recs), np.std(recs)
  f1_mean, f1_std = np.mean(f1s), np.std(f1s)
  roc_mean, roc_std = np.mean(rocs), np.std(rocs)
  pra_mean, pra_std = np.mean(pras), np.std(pras)
  mcc = matthews_corrcoef(y_val_all, y_pred_all)

  composite = (roc_mean + pra_mean + acc_mean) / 3

  log(
    f"  ✅  Acc:{{acc_mean:.4f}}±{{acc_std:.4f}} | "
    f"Precision:{{prec_mean:.4f}}±{{prec_std:.4f}}  |"
    f"Recall:{{rec_mean:.4f}}±{{rec_std:.4f}} |"
    f"F1:{{f1_mean:.4f}}±{{f1_std:.4f}} | "
    f"AUROC:{{roc_mean:.4f}}±{{roc_std:.4f}} | "
    f"PRAUC:{{pra_mean:.4f}}±{{pra_std:.4f}} | "
    f"MCC:{{mcc:.4f}} | "
    f"Composite:{{composite:.4f}}"
  )

  iter_time = time.perf_counter() - _iter_t0
  if composite > best_score:
    best_score = composite
    best_result = {{
        'tag': tag,
        'composite': composite,
        'roc': f"{{roc_mean:.4f}}±{{roc_std:.4f}}",
        'pra': f"{{pra_mean:.4f}}±{{pra_std:.4f}}",
        'acc': f"{{acc_mean:.4f}}±{{acc_std:.4f}}",
        'f1': f"{{f1_mean:.4f}}±{{f1_std:.4f}}",
        'prec': f"{{prec_mean:.4f}}±{{prec_std:.4f}}",
        'rec': f"{{rec_mean:.4f}}±{{rec_std:.4f}}",
        'params': params,
        'execution_time': iter_time,
        'dataset_size': DATASET_SIZE,
        'kernel': QKERNEL,
        'acc_mean': float(acc_mean),
        'acc_std': float(acc_std),
        'n_optimal': n_optimal,
    }}

  done_configs[tag] = {{"composite": composite, "params": params}}
  save_checkpoint(ckpt_path, {{
      "done_configs": done_configs,
      "results": results,
      "best_result": best_result,
      "best_score": best_score,
  }})

log(f"\\n🏆 Best config : {{best_result['tag']}}")
log(
    f"   Composite   : {{best_result['composite']}} "
    f"(AUROC={{best_result['roc']}} | "
    f"PRAUC={{best_result['pra']}} | "
    f"Acc={{best_result['acc']}} |"
    f"Prec={{best_result['prec']}} |"
    f"Rec={{best_result['rec']}}) |"
)

# ── Collect best result ──
best_result['model'] = f"QSVC ({{QKERNEL}}) @ {{DATASET_SIZE}}"
scalability_results[(DATASET_SIZE, QKERNEL)] = dict(best_result)
log(f"✅ [QSVC ({{QKERNEL}}) @ {{DATASET_SIZE}}] recorded | Exec. time: {{best_result['execution_time']:.1f}}s")
'''


TABLE_CELL = '''# Tabel ringkasan
scal_rows = []
for (size, kernel), r in sorted(scalability_results.items()):
    scal_rows.append({
        'Dataset Size': size,
        'Kernel': kernel,
        'Accuracy': r.get('acc', f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}"),
        'Precision': r.get('prec', 'N/A'),
        'Recall': r.get('rec', 'N/A'),
        'F1-Score': r.get('f1', 'N/A'),
        'ROC-AUC': r.get('roc', 'N/A'),
        'PR-AUC': r.get('pra', 'N/A'),
        'n_optimal (PCA)': r.get('n_optimal', 'N/A'),
        'Exec. Time (s)': round(r['execution_time'], 2) if 'execution_time' in r else 'N/A',
    })
scal_df = pd.DataFrame(scal_rows)
display(scal_df)
'''


PLOT_CELL = '''# Plot Akurasi vs Jumlah Data
import matplotlib.pyplot as plt

sizes_sorted = sorted(scalability_datasets.keys())
kernel_styles = {
    'custom': {'marker': 'o', 'color': '#1f77b4'},
    'full':   {'marker': 's', 'color': '#d62728'},
}

plt.figure(figsize=(9, 6))
for kernel in ['custom', 'full']:
    xs, ys, yerr = [], [], []
    for size in sizes_sorted:
        if (size, kernel) in scalability_results:
            r = scalability_results[(size, kernel)]
            xs.append(size)
            ys.append(r['acc_mean'])
            yerr.append(r['acc_std'])
    if not xs:
        continue
    style = kernel_styles.get(kernel, {'marker': '^'})
    plt.errorbar(
        xs, ys, yerr=yerr,
        marker=style['marker'],
        color=style.get('color'),
        capsize=4, linewidth=2, markersize=8,
        label=f"QSVC ({kernel})",
    )
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.xscale('log')
plt.xticks(sizes_sorted, [str(s) for s in sizes_sorted])
plt.xlabel("Jumlah Data")
plt.ylabel("Akurasi (5-fold CV mean)")
plt.title("Akurasi QSVC vs Jumlah Data")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_dir = Path("./results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "qsvc_accuracy_vs_dataset_size.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"📊 Plot disimpan di {fig_path}")
plt.show()
'''


# ── Locate scalability section start (markdown header "ee6ea197") ──
idx_header = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "ee6ea197")

# Remove everything after the markdown header
nb["cells"] = nb["cells"][: idx_header + 1]

# Build new cells: config + (### Custom + 4 cells) + (### Full + 4 cells) + table + plot
new_cells = [code(CONFIG_CELL)]

SIZES = [500, 1000, 5000, 10000]

for KERNEL in ["custom", "full"]:
    new_cells.append(md(f"### {KERNEL.capitalize()}"))
    for SIZE in SIZES:
        new_cells.append(code(cell_for(SIZE, KERNEL)))

new_cells.append(md("### 📊 Ringkasan & Plot"))
new_cells.append(code(TABLE_CELL))
new_cells.append(code(PLOT_CELL))

nb["cells"].extend(new_cells)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK — wrote {len(new_cells)} new cells (total {len(nb['cells'])})")
