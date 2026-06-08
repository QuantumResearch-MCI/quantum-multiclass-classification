# Tahap 1 — Split Dataset & Eksplorasi Awal

Notebook: [`1.1_split_dataset`](../notebooks/1.1_split_dataset.ipynb),
[`1.2_split_stratify`](../notebooks/1.2_split_stratify.ipynb),
[`1.3_split_stratify_classical`](../notebooks/1.3_split_stratify_classical.ipynb)

## Tujuan
Menyiapkan data untuk eksperimen: membuat split train/val/test dan menjalankan baseline
evaluasi awal lintas `random_state`.

## 1.1 — Strategi Split
Membandingkan beberapa cara memisahkan data:
- **StratifiedShuffleSplit** dua tahap (70 % train, 15 % val, 15 % test) yang **menjaga proporsi
  kelas** `Kategori`; hasil disimpan ke `dataset/StratifiedShuffleSplit/`.
- **Split manual** berbasis index acak (`np.random.default_rng`) dengan rasio 70/15/15 untuk
  beberapa `random_state`, disimpan ke `dataset/manual/{state}/`.
- **Subsampling**: `train_test_split` stratified (≈300 baris) dan `groupby('Kategori').sample(n=3)`
  untuk dataset kecil/uji cepat (`reduced_dataset.csv`).

## 1.2 — Stratify (semua model termasuk quantum)
Notebook eksperimen baseline:
- Setup environment: `pip install` (qiskit-aer/algorithms/ML/ibm-runtime, xgboost, catboost,
  libsvm, dll.), `sys.path` ke root, `reload_package`, suppress warnings.
- Konfigurasi 12 fitur sensor + target `Kategori`; daftar kernel klasik (`linear, poly, rbf, sigmoid`)
  dan quantum (`full, linear, circular, pauli_x/y/z`).
- EDA: `data.head()`, histogram `Kategori` (memperlihatkan imbalance).
- **Evaluasi lintas `random_state` `[10, 42, 100, 2021, 1234]`** memakai `prepare_data` +
  fungsi `evaluate_svc / evaluate_xgboost / evaluate_catboost / evaluate_quantum / evaluate_libsvm`.
- Agregasi hasil (`groupby(model, kernel)` → mean/std accuracy, F1, val-acc, waktu), lalu
  `aggregate_results`, `report`, dan **grand confusion matrix** per model–kernel.

## 1.3 — Stratify (klasik saja)
Identik dengan 1.2 tetapi **blok `evaluate_quantum` dinonaktifkan** (komentar) sehingga hanya
SVC, XGBoost, CatBoost, dan LibSVM yang dievaluasi — versi cepat tanpa beban simulasi quantum.

## Catatan Imbalance & CV
- Histogram `Kategori` di sini adalah titik awal yang menunjukkan dominasi kelas E dan minoritas B.
- Strategi split sudah **stratified** sejak tahap ini agar setiap subset merepresentasikan semua kelas.
- Belum ada penanganan grup di tahap ini — baris diperlakukan independen (lihat tahap 3.4.1 untuk
  koreksi kebocoran ber-grup dengan **SGKF**).
