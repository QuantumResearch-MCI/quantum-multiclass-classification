# Ringkasan Notebook — Klasifikasi Mutu Teh Hijau (Klasik vs Quantum)

Dokumen ini merangkum seluruh notebook eksperimen pada folder [`notebooks/`](../notebooks),
disusun runtut dari tahap **1 sampai 7**. Tujuan proyek: membandingkan model machine
learning **klasik** (SVC, XGBoost, CatBoost, MLP, CNN1D, KernelKNN) dengan model
**quantum / hybrid quantum** (QSVC, QXGB, QCat) pada **Dataset Teh Hijau** untuk
memprediksi `Kategori` mutu (A–E).

## Peta Notebook (1 → 7)

| Tahap | Notebook | Isi | Detail |
|-------|----------|-----|--------|
| **1** | `1.1`–`1.3` split | Strategi pemisahan dataset & eksplorasi awal | [01_split_dataset.md](01_split_dataset.md) |
| **2** | `2_main` | Pipeline evaluasi utama (CV stratified, ROC/PR/CM) | [02_main_evaluation.md](02_main_evaluation.md) |
| **3** | `3.1`–`3.7` HPO | Hyperparameter tuning klasik, DL, dan quantum (FSK/PQK/FQK-hardware) | [03_hyperparameter_optimization.md](03_hyperparameter_optimization.md) |
| **4** | `4_qsvc_iqp_cust_comparison` | Skalabilitas QSVC kernel `custom` vs `full` | [04_scalability_qsvc.md](04_scalability_qsvc.md) |
| **5** | `5.1`–`5.3` compare | Perbandingan kernel klasik & kernel quantum FSK vs PQK | [05_kernel_comparison.md](05_kernel_comparison.md) |
| **6** | `6.1`–`6.2` overfit | Diagnostik overfit/underfit, barren plateau, per-kelas | [06_overfit_diagnostics.md](06_overfit_diagnostics.md) |
| **7** | `7.feature_correlation` | Korelasi fitur–target (Spearman) | [07_feature_correlation.md](07_feature_correlation.md) |

---

## Dataset & Ketidakseimbangan Kelas (Data Imbalance)

`dataset/Dataset_TehHijau.csv` berisi **10.409 baris × 130 kolom**. Fitur intinya:

- **12 fitur sensor e-nose**: `MQ3, TGS822, TGS2602, MQ5, MQ138, TGS2620, TGS813, TGS2600, TGS2611, TGS2603, Humidity, Celsius`
- **5 fitur sensorik (organoleptik)**: `Aroma, Taste, Color, Appearance, Dreg`
- **Fitur statistik turunan** per sensor (`*_mean, *_min, *_max, *_std, *_skew, *_kurtosis, *_auc`)
- **Kolom grup**: `Chop_ID` (69 grup), `Sampling_ID` (274 grup)
- **Target**: `Kategori` (A–E) dan `Standar Kualitas` (Baik / Cacat Mutu)

### Distribusi `Kategori` — sangat timpang

| Kategori | Jumlah | Proporsi |
|----------|--------|----------|
| A | 927 | 8,9 % |
| **B** | **229** | **2,2 %** ← kelas minoritas |
| C | 2.287 | 22,0 % |
| D | 2.317 | 22,3 % |
| E | 4.649 | 44,7 % ← kelas mayoritas |

Rasio mayoritas:minoritas (E:B) ≈ **20 : 1**. `Standar Kualitas` juga timpang (Baik 7.504 vs Cacat Mutu 2.905 ≈ 2,6 : 1).

### Dampak imbalance & cara penanganannya di proyek ini

Imbalance membuat metrik **accuracy** menyesatkan (model bisa "menang" hanya dengan menebak E)
dan kelas minoritas B mudah terabaikan. Mitigasi yang diterapkan:

1. **Metrik majemuk, bukan accuracy saja.** Skor seleksi HPO memakai
   `(AUROC + PRAUC + Accuracy) / 3`, dan semua metrik dihitung `average='weighted'`
   (F1, precision, recall, AUROC, PRAUC) plus **MCC** dan **balanced accuracy** pada diagnostik.
2. **Stratifikasi** di semua split & fold supaya proporsi A–E terjaga di tiap lipatan.
3. **Class weighting**:
   - SVC: `class_weight='balanced'` (lihat tahap 3.4).
   - XGBoost: `compute_sample_weight('balanced')` dijadikan `sample_weight` saat `fit`.
4. **Diagnostik per-kelas** (notebook 6.1 cell 4): confusion matrix + F1 per kelas +
   `balanced_accuracy_score` untuk melihat **kelas mana** yang gagal, bukan hanya rata-rata.

---

## StratifiedKFold (SKF) vs StratifiedGroupKFold (SGKF)

Ini adalah keputusan metodologis paling penting di proyek ini.

### StratifiedKFold (SKF) — dipakai di mayoritas notebook

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    ...
```

- **Apa**: membagi data jadi 5 lipatan sambil **menjaga proporsi kelas** A–E di tiap lipatan.
- **Kapan dipakai**: tahap 2, 3.1–3.3, 3.5–3.7, 4, 5 — yaitu saat fitur yang dipakai adalah
  fitur sensor e-nose dan setiap baris diperlakukan independen.
- **Kelemahan pada dataset ini**: SKF **tidak sadar grup**. Karena banyak baris berasal dari
  **`Sampling_ID`/`Chop_ID` yang sama** (pengukuran berulang dari sampel/chop yang sama),
  baris-baris berkorelasi tinggi bisa tersebar ke train **dan** validasi sekaligus →
  **kebocoran data ber-grup (group leakage)**. Akibatnya metrik CV **menggelembung**
  (over-optimistis) dan tidak mencerminkan generalisasi ke sampel/chop baru.

### StratifiedGroupKFold (SGKF) — koreksi kebocoran di notebook 3.4.1 & 3.4.2

```python
groups = data["Sampling_ID"].values        # cell 9
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y, groups):   # groups wajib dilewatkan
    ...
```

- **Apa**: tetap menjaga proporsi kelas, **tetapi** menjamin **satu grup tidak pernah muncul
  di train dan validasi sekaligus**. Validasi dilakukan terhadap grup yang benar-benar baru.
- **Kenapa `Sampling_ID`, bukan `Chop_ID`** (catatan di kode): kelas B hanya punya **2 `Chop_ID`**,
  sehingga 5-fold mustahil distratifikasi per chop; sedangkan `Sampling_ID` untuk B ada **9 grup**,
  masih cukup untuk dibagi 5 lipatan. (Jumlah grup per kelas — Chop_ID: A=4, B=2, C=14, D=13, E=36;
  Sampling_ID: A=20, B=9, C=58, D=55, E=132.)
- **Konsekuensi**: skor SGKF biasanya **lebih rendah tetapi lebih jujur** daripada SKF. Selisih
  antara keduanya adalah estimasi besarnya kebocoran. Notebook 3.4.1 juga mengganti fitur ke
  atribut sensorik (`Aroma, Taste, Color, Appearance, Dreg`) untuk evaluasi yang lebih bersih.

> **Ringkas:** **hanya notebook bernama `*_StratifiedGroupKFold` (3.4.1 & 3.4.2)** yang memakai
> SGKF; **semua notebook lain memakai SKF**. SGKF adalah jawaban langsung atas isu
> *grouped-data-leakage* pada Dataset Teh Hijau.

---

## Apa yang Sudah Diimplementasikan (Overall)

- **Eksplorasi & split**: StratifiedShuffleSplit, split manual, subsampling, histogram kelas, `describe()`.
- **Pipeline standar** di semua model: `StandardScaler → PCA (≥95 % / 92 % varians) → estimator`.
- **Model klasik**: SVC (linear/poly/rbf/sigmoid), XGBoost (gbtree/dart), CatBoost, MLP, CNN1D, KernelKNN.
- **Model quantum/hybrid**: QSVC, QXGB (gbtree/dart), QCat dengan tiga mode kernel:
  - **FSK** (Fidelity Statevector Kernel) — simulasi statevector.
  - **PQK** (Projected Quantum Kernel) — proyeksi observable + RBF klasik.
  - **FQK-hardware** — fidelity kernel pada backend nyata IBM Quantum (`qiskit-ibm-runtime`).
  - Variasi feature map: `full, linear, circular, pauli_x/y/z`, plus circuit kustom (`custom_*`).
- **HPO**: grid manual via `itertools.product` (+ opsi `BayesSearchCV`) dengan **checkpointing**
  (resume `.pkl`) dan **logging** per-section ke file.
- **Penanganan imbalance**: stratifikasi, `class_weight='balanced'`, `compute_sample_weight`, metrik majemuk.
- **Validasi**: **StratifiedKFold** di mayoritas notebook; **StratifiedGroupKFold** khusus di
  notebook bernama `*_StratifiedGroupKFold` (3.4.1 & 3.4.2) untuk mencegah kebocoran ber-grup.
- **Skalabilitas**: evaluasi QSVC pada ukuran 500 / 1k / 5k / 10k baris.
- **Diagnostik kernel**: Gram-matrix (overfit/underfit), barren plateau (sweep `n_qubits`),
  learning curve, dan diagnostik per-kelas.
- **Analisis fitur**: heatmap korelasi Spearman fitur–target, dengan peringatan eksplisit agar
  `Chop_ID`/`Sampling_ID` **tidak dipakai sebagai fitur** (terkait kebocoran).

> Catatan teknis: hampir semua notebook diawali `pip install` dependensi, load `.env`
> (`PROJECT_ROOT`, `IBMQ_TOKEN`), helper `reload_package`, dan suppress warnings. Folder
> `results/checkpoints` harus dihapus bila dataset/search-space berubah, agar konfigurasi baru
> benar-benar dieksekusi (bukan memuat hasil run lama).
