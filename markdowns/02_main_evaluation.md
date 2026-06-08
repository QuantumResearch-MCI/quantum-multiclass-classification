# Tahap 2 — Pipeline Evaluasi Utama

Notebook: [`2_main`](../notebooks/2_main.ipynb)

## Tujuan
Pipeline evaluasi terpadu yang menjalankan beberapa model dalam satu loop cross-validation,
lengkap dengan kurva ROC, PR-AUC, confusion matrix, dan perbandingan performa.

## Alur Implementasi
1. **Setup**: `pip install` (+ `scikit-optimize`, `python-dotenv`), load `.env`
   (`IBMQ_TOKEN`), `reload_package`, suppress warnings.
2. **Data**: muat `Dataset_TehHijau.csv`, 12 fitur sensor, target `Kategori`.
   - EDA: `head`, `describe`, histogram `Kategori` (imbalance), cek kelas unik.
3. **PCA**: fungsi `plot_pca_variance` menentukan `n_optimal` komponen untuk menahan ≥95 % varians
   (visualisasi cumulative explained variance). PCA final bisa diaktifkan/nonaktifkan.
4. **Konfigurasi pipeline**: `LabelEncoder` untuk target; model didefinisikan dalam dict
   (`SVC_*`, `XGB_gbtree`, dll.; QSVC/QESVC/QXGB/LibSVM dengan `mode='fsk'` untuk quantum).
   Tiap model dibungkus `Pipeline` (scaler → estimator).
5. **Cross-validation (SKF)**:
   ```python
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
       ...
   ```
   Mengumpulkan per-model: accuracy, precision, recall, F1, confusion_matrix, `y_true`, `y_prob`.
6. **Agregasi & visualisasi**:
   - `aggregate_results` → tabel metrik + matriks confusion gabungan.
   - `plot_roc_curve`, `plot_prauc` (penting untuk data imbalanced), `plot_conf_matrix`,
     `plot_performance_comparison`, `report`.

## Catatan Imbalance & CV
- Memakai **StratifiedKFold** (baris dianggap independen) — menjaga proporsi A–E tiap fold,
  tetapi **belum** menangani kebocoran ber-grup (`Sampling_ID`).
- **ROC dan PR-AUC** ditambahkan secara eksplisit karena lebih informatif daripada accuracy pada
  kelas timpang; confusion matrix gabungan membantu melihat kebingungan antar kelas A–E.
