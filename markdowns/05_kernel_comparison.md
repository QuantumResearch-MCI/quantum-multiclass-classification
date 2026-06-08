# Tahap 5 — Perbandingan Kernel

Notebook: [`5.1_compare_classical_kernel`](../notebooks/5.1_compare_classical_kernel.ipynb),
[`5.2_compare_fsk_pqk`](../notebooks/5.2_compare_fsk_pqk.ipynb),
[`5.3_compare_fsk_pqk`](../notebooks/5.3_compare_fsk_pqk.ipynb)

## 5.1 — Perbandingan Kernel Klasik
Membandingkan kernel pada model klasik:
- **SVC**: linear, poly, rbf, sigmoid.
- **KernelKNN**: kernel RBF dan polynomial.
- Diakhiri **Tabel Evaluasi Akhir**.
- Catatan: salah satu run CatBoost memakai `loss_function="Multiclass"` (waktu ≈168,9 s),
  dua lainnya `loss_function="MulticlassOneVsAll"`.

## 5.2 & 5.3 — Perbandingan Kernel Quantum: FSK vs PQK
Kedua notebook hampir identik (refinement berurutan), fokus membandingkan **mode kernel quantum**
pada QSVC lewat kerangka skalabilitas yang sama seperti tahap 4:
- Ukuran dataset 500 / 1k / 5k / 10k, `scalability_params = {'C': 1, 'lambda_': 0.3}`,
  **StratifiedKFold 5-fold**.
- `run_scalability(QKERNEL, mode)` kini menerima argumen **`mode`** (`'fsk'` atau `'pqk'`),
  sehingga bisa membandingkan **FSK vs PQK** untuk kernel yang sama.
- `plot_scalability_comparison(*runs)` memplot **N hasil run sekaligus** dalam satu figure
  (6 metrik standar).
- Eksekusi aktif:
  ```python
  full_pqk = run_scalability('full', 'pqk')
  full_fsk = run_scalability('full', 'fsk')
  plot_scalability_comparison(full_pqk, full_fsk)
  ```
  (baris untuk kernel `custom_*`, `linear`, `circular` tersedia tapi dikomentari.)

> Perbedaan 5.2 vs 5.3 hanya pada iterasi/penyempurnaan; secara fungsional sama —
> membandingkan **FSK** dan **PQK** pada kernel `full` lintas ukuran dataset.

## Catatan Imbalance & CV
- **SKF** dengan stratifikasi di semua perbandingan.
- Metrik majemuk + 6 metrik (termasuk PRAUC, F1, recall) ditampilkan agar perbandingan kernel
  adil pada kelas timpang, bukan hanya berdasarkan accuracy.
