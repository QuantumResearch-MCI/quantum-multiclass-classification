# Tahap 3 — Hyperparameter Optimization (HPO)

Notebook: `3.1`–`3.7`. Inilah tahap inti tuning model klasik, deep learning, dan quantum.

| Notebook | Fokus | CV | Catatan |
|----------|-------|----|---------|
| [`3.1_hpo_classical_all`](../notebooks/3.1_hpo_classical_all.ipynb) | HPO klasik (SVC, XGBoost, CatBoost) | SKF | Grid penuh |
| [`3.2_hpo_classical_all_dl`](../notebooks/3.2_hpo_classical_all_dl.ipynb) | HPO deep learning (MLP, CNN1D) | SKF | Keras/TensorFlow |
| [`3.3_hpo_classical_best_param`](../notebooks/3.3_hpo_classical_best_param.ipynb) | Re-run dengan param terbaik | SKF | `class_weight=None` |
| [`3.4_hpo_classical_best_param_balanced`](../notebooks/3.4_hpo_classical_best_param_balanced.ipynb) | Param terbaik **+ penanganan imbalance** | SKF | `class_weight='balanced'`, `compute_sample_weight` |
| [`3.4.1_..._StratifiedGroupKFold`](../notebooks/3.4.1_hpo_classical_best_param_StratifiedGroupKFold.ipynb) | Validasi **bersih** anti-kebocoran | **SGKF** | fitur sensorik + `groups=Sampling_ID` |
| [`3.5_hpo_quantum_fsk`](../notebooks/3.5_hpo_quantum_fsk.ipynb) | HPO quantum **FSK** | SKF | `mode='fsk'` |
| [`3.6_hpo_quantum_pqk`](../notebooks/3.6_hpo_quantum_pqk.ipynb) | HPO quantum **PQK** | SKF | `mode='pqk'` |
| [`3.7_hpo_quantum_fqk_hardware`](../notebooks/3.7_hpo_quantum_fqk_hardware.ipynb) | HPO quantum **hardware** IBM | SKF | `mode='fqk-hardware'` |

## Pola umum semua notebook HPO
- Setup: `pip install`, load `.env` (`PROJECT_ROOT`), `reload_package`, suppress warnings.
- Data + PCA (`plot_pca_variance`, threshold 0,92–0,95) → `n_optimal` komponen.
- **Logger** per-section ke `results/logs/...` dan **checkpointing** (`.pkl`, resume otomatis,
  simpan atomik via `.tmp` → rename). Hapus folder checkpoint bila konfigurasi berubah.
- Search space didefinisikan sebagai dict, dieksplorasi dengan `itertools.product`
  (opsi `BayesSearchCV` dari `scikit-optimize` tersedia).
- **Skor seleksi: `(AUROC + PRAUC + Accuracy) / 3`** — sengaja majemuk agar adil pada kelas timpang.
- Pipeline: `StandardScaler → PCA → estimator`; metrik per-fold lengkap
  (Acc, Prec, Rec, F1 weighted, AUROC, PRAUC, MCC).

## 3.1 Klasik
- **SVC**: linear / poly / rbf / sigmoid; grid `C, class_weight∈{balanced,None}, tol, gamma, degree, coef0`.
- **XGBoost**: gbtree & dart; grid `n_estimators, learning_rate, max_depth` (subsample/colsample opsional).
- **CatBoost** juga di-tune. Output CSV per model di `results/`.

## 3.2 Deep Learning
- **MLP** dan **CNN1D** (Keras), seed di-fix (`set_random_seed(42)`), tetap di dalam **SKF** 5-fold.
- Tabel evaluasi akhir DL terpisah.

## 3.3 / 3.4 / 3.4.1 — tiga perlakuan imbalance & kebocoran
Ketiganya memakai **parameter terbaik hasil 3.1**, perbedaannya pada penanganan imbalance/grup:

- **3.3** — `class_weight=None` (baseline tanpa penyeimbangan).
- **3.4 (balanced)** — penanganan imbalance:
  - SVC: `class_weight` dihapus dari grid dan di-set `'balanced'` di model.
  - XGBoost: `from sklearn.utils.class_weight import compute_sample_weight` → `sample_weight`
    saat `fit` (komentar di kode: *"← imbalance handling"*).
- **3.4.1 (StratifiedGroupKFold)** — koreksi **kebocoran ber-grup**:
  ```python
  groups = data["Sampling_ID"].values   # B punya 9 Sampling_ID (cukup 5-fold),
                                         # sedangkan Chop_ID untuk B hanya 2 (tidak cukup)
  skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  for train_idx, val_idx in skf.split(X, y, groups):
      model = Pipeline([('scaler', StandardScaler()),
                        ('pca', PCA(n_optimal)),
                        ('svc', SVC(class_weight='balanced', kernel='linear',
                                    probability=True, decision_function_shape='ovr', ...))])
  ```
  - **Fitur diganti** ke atribut sensorik: `Aroma, Taste, Color, Appearance, Dreg`.
  - Menjamin grup `Sampling_ID` tidak bocor antara train & validasi → metrik lebih jujur.
  - Tetap `class_weight='balanced'` untuk imbalance.
  - ℹ️ Hanya notebook bernama `*_StratifiedGroupKFold` (3.4.1 & 3.4.2) yang memakai SGKF;
    notebook lain memakai StratifiedKFold.

## 3.5 / 3.6 / 3.7 — Quantum
Struktur identik (≈158–160 sel), beda di **`mode`**:
- **3.5 `fsk`** — Fidelity Statevector Kernel (simulasi statevector).
- **3.6 `pqk`** — Projected Quantum Kernel (proyeksi observable + RBF klasik).
- **3.7 `fqk-hardware`** — fidelity kernel di **hardware IBM Quantum**:
  `QiskitRuntimeService.save_account(token=IBMQ_TOKEN)`; search space quantum menambah
  `lambda_` dan `n_measurements` (mis. 256 untuk non-fsk).

Setiap notebook quantum men-tune **QSVC**, **QXGB** (gbtree & dart), dan **QCat**, untuk tiap
feature map (`full, linear, circular, pauli_x/y/z`). Hasil terbaik dikumpulkan ke `all_best`.

## Catatan Imbalance & CV
- 3.1–3.3, 3.5–3.7 memakai **SKF** (baris independen).
- **3.4** menambah class weighting; **3.4.1** (dan **3.4.2**) naik level ke **SGKF** untuk mengatasi
  kebocoran ber-grup yang membuat skor SKF over-optimistis. Lihat penjelasan lengkap di
  [README.md](README.md).
