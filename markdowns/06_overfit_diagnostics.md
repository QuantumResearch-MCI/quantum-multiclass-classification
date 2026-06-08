# Tahap 6 — Diagnostik Overfit/Underfit & Barren Plateau

Notebook: [`6.1_check_kernel_overfit`](../notebooks/6.1_check_kernel_overfit.ipynb) (Dataset Teh Hijau),
[`6.2_check_kernel_overfit_iris`](../notebooks/6.2_check_kernel_overfit_iris.ipynb) (dataset Iris, sanity check)

## Tujuan
Alat bantu **desain sirkuit quantum kustom**: mendeteksi apakah kernel quantum mengalami overfit
(memorisasi), underfit, atau **barren plateau / kernel concentration**.

## Empat Diagnostik
1. **Kernel matrix diagnostic** (cell 18) — bangun Gram matrix `K(X,X)` lalu cek:
   - **Overfit**: `K ≈ identity` (diagonal dominan, semua sampel dianggap berbeda → memorisasi).
   - **Underfit**: entry collapse / Kernel-Target Alignment (KTA) rendah.
   - **Sehat**: struktur blok per kelas terlihat jelas.
   - Konfigurasi bisa diganti: `KERNEL_KIND` (`full / linear / circular / custom_*`),
     `MODE` (`fsk / fqk / pqk`), `LAMBDA_`, `DATASET_SIZE`.
2. **Barren plateau diagnostic** (cell 19) — sweep `n_qubits` dan amati scaling `Var(K_off_diag)`.
   Penurunan eksponensial (`var ~ exp(-c·n_qubits)`, garis lurus turun pada skala log) menandakan
   **kernel concentration / barren plateau**; kernel sehat menjaga var ~ O(1). Membandingkan
   beberapa sirkuit: `custom_full/linear/circular` dan `custom_rbf_full/linear/circular`.
3. **Model overfit/underfit** (cell 20) — `learning_curve` standar untuk pipeline QSVC
   (`StandardScaler → PCA → QSVC`), menilai gap train vs validasi.
4. **Per-class diagnostic** (cell 21) — confusion matrix, `classification_report`,
   **F1 per kelas**, dan `balanced_accuracy_score` untuk **mengidentifikasi kelas mana yang gagal** —
   esensial pada dataset imbalanced (kelas B yang minoritas paling rawan).

## 6.2 — Versi Iris
Notebook kembar yang menjalankan diagnostik yang sama pada dataset **Iris** sebagai
kontrol/sanity-check (data seimbang & terpisah baik), untuk memastikan alat diagnostik bekerja
benar sebelum diterapkan ke Dataset Teh Hijau.

## Catatan Imbalance & CV
- Diagnostik per-kelas dan `balanced_accuracy` dirancang khusus untuk kondisi **imbalance**.
- Split memakai `train_test_split(..., stratify=y)`; fokus di sini adalah perilaku kernel,
  bukan estimasi generalisasi (kebocoran ber-grup dibahas di tahap 3.4.1).
