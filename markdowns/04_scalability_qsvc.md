# Tahap 4 — Skalabilitas QSVC (Kernel `custom` vs `full`)

Notebook: [`4_qsvc_iqp_cust_comparison`](../notebooks/4_qsvc_iqp_cust_comparison.ipynb)

## Tujuan
Mengukur bagaimana performa **QSVC** berubah terhadap ukuran dataset, membandingkan feature map
**`custom`** (sirkuit rancangan sendiri) dengan **`full`** (gaya IQP/ZZFeatureMap entanglement penuh).

## Implementasi
- Setup standar (`pip install`, `.env` `PROJECT_ROOT`, `reload_package`), `plot_pca_variance`,
  logger ke `results/logs/fsk`, checkpoint ke `results/checkpoints/scalability`.
- **Empat ukuran dataset**:
  ```python
  scalability_datasets = {
      500:   "Dataset_TehHijau500.csv",
      1000:  "Dataset_TehHijau1k.csv",
      5000:  "Dataset_TehHijau5k.csv",
      10000: "Dataset_TehHijau.csv",   # full
  }
  scalability_kernels = ['custom', 'full']
  scalability_params  = {'C': 1, 'lambda_': 0.3}
  ```
- `run_scalability(QKERNEL)` mengevaluasi QSVC pada tiap ukuran dengan **StratifiedKFold 5-fold**,
  metrik lengkap per-fold (Acc, Prec, Rec, F1, AUROC, PRAUC, MCC), skor majemuk
  `(AUROC + PRAUC + Acc) / 3`, dan checkpoint per iterasi ukuran.
- `plot_scalability_comparison('custom', 'full')` memplot 6 metrik (mean ± std) terhadap ukuran
  dataset; hasil disimpan ke `results/plots/scalability_comparison.png`.
- Eksekusi: `run_scalability('custom')`, `run_scalability('full')`, lalu plot.

## Catatan Imbalance & CV
- **SKF** (baris independen) dengan stratifikasi kelas pada setiap ukuran dataset.
- Fokus notebook adalah **tren skalabilitas** (apakah keunggulan kernel quantum bertahan saat data
  membesar), bukan penanganan kebocoran ber-grup.
