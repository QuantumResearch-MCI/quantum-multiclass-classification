# Handle Class Imbalance — Resampling vs Adjusted Model

Skenario penanganan *class imbalance* pada `dataset/Dataset_TehHijau.csv`
lewat **resampling data**, dibandingkan dengan **adjusted model** (pendekatan
kemarin: `class_weight='balanced'`, `gamma='scale'`, dst.).

| Notebook | Skenario | Teknik |
|---|---|---|
| [oversampling_smote.ipynb](oversampling_smote.ipynb) | Oversampling | `SMOTE` |
| [undersampling_random.ipynb](undersampling_random.ipynb) | Undersampling | `RandomUnderSampler` |
| [custom_sampling.ipynb](custom_sampling.ipynb) | Custom (target per-kelas) | `make_custom_resampler` |
| [hybrid_sampling.ipynb](hybrid_sampling.ipynb) | Hybrid (multi-teknik) | `SMOTEENN`, `SMOTETomek` |

`hybrid_sampling.ipynb` memakai struktur **multi-teknik**: tiap model diadu untuk
adjusted + **tiap** resampler di `RESAMPLERS` sekaligus (tabel best-per-model,
confusion-matrix & learning-curve per teknik). Pola ini mudah dipakai ulang untuk
membandingkan banyak metode lain (lihat di bawah).

### Ragam metode resampling (imbalanced-learn)
- **Oversampling:** `RandomOverSampler` (ROS), `SMOTE`, `BorderlineSMOTE`, `ADASYN`.
- **Undersampling:** `RandomUnderSampler` (RUS), `TomekLinks`, `EditedNearestNeighbours` (ENN).
- **Hybrid:** `SMOTEENN`, `SMOTETomek`.

Tidak ada yang "terbaik" universal — bandingkan empiris pakai balanced-accuracy /
macro-F1 + learning curve. Untuk menambah teknik ke notebook hybrid (atau membuat
notebook over/under multi-teknik), cukup isi `RESAMPLERS = {nama: resampler, ...}`.

### Custom sampling (target per-kelas)
`custom_sampling.ipynb` membiarkan kamu set target jumlah **per kelas** lewat
`CUSTOM_TARGETS` di sel Konfigurasi. Nilai boleh `int` (jumlah absolut) atau nama
kelas lain (samakan jumlahnya). Kelas yang tak disebut **tidak diubah**. Target
< jumlah → undersample (Random); target > jumlah → oversample (SMOTE). Target
dihitung **per fold**, jadi `{'B': 'C', 'E': 'C'}` (default: B & E ke jumlah C,
A/C/D tetap) tetap benar walau jumlah C beda antar fold. Lihat
[`make_custom_resampler`](../../utils/imbalance_eval.py).

## Cara membandingkan
Tiap notebook menjalankan **7 model identik** (SVC linear/poly/rbf/sigmoid,
XGBoost gbtree/dart, CatBoost — best params dari `3.4.2_..._newhpobalanced.ipynb`)
dalam **dua kondisi** dengan split & hyperparameter sama:

- **adjusted** — tanpa resampling, classifier pakai bobot kelas `balanced`.
- **resample** — resampler skenario di fold training, classifier **tanpa** bobot
  (agar efek murni resampling yang terukur).

Selisih (`Delta = resample − adjusted`) untuk `balanced_accuracy`, `macro_f1`,
`roc_auc`, `mcc` ditampilkan di tabel berpasangan + confusion matrix per kelas.

**Catatan penting:** di arm **resample**, model **tidak** memakai `class_weight`/
`sample_weight`/`auto_class_weights` `balanced` — karena imbalance sudah ditangani
di level data oleh resampling (tidak perlu di-balance dua kali). Arm **adjusted**
(balanced, tanpa resampling) hanya dipakai sebagai pembanding baseline.

### Learning curve (overfit/underfit)
Tiap notebook punya section **learning curve** (train vs val `balanced_accuracy`
pada beberapa ukuran training set) untuk model mode resample. `gap = train − val`
besar → overfit; val rendah → underfit. SVC pakai `probability=False` di sini
agar lebih cepat (LC hanya butuh `.predict()`).

## Catatan CV (dataset ber-grup)
274 `Sampling_ID` menghasilkan ~38 pembacaan sensor mirip per sampel. Maka:
- Split: `StratifiedKFold` (sesuai keputusan proyek). ⚠️ Grup `Sampling_ID` **tidak**
  dijaga, jadi pembacaan dari sampel yang sama bisa tersebar ke train & validasi →
  metrik cenderung over-optimistis (kebocoran ber-grup diterima secara sadar).
- Resampling **hanya** di fold training (dijamin `imblearn.Pipeline`); fold
  validasi selalu distribusi asli. Tidak ada SMOTE yang bocor ke validasi.

Logika CV terpusat di [`utils/imbalance_eval.py`](../../utils/imbalance_eval.py).

## Output
Tiap notebook menyimpan ke `results/<strategy>/`:
`*_composition.csv`, `*_composition_before_after.png`, `*_metrics_long.csv`,
`*_paired_delta.csv`, `*_bar_compare.png`, `*_confusion_matrices.png`,
`*_learning_curve.png`, `*_run.log`.

### Komposisi sebelum vs sesudah sampling (section 4)
Histogram jumlah kelas **sebelum vs sesudah** resampling + scatter PCA (PC1–PC2)
untuk melihat di mana titik sintetis muncul. Dihitung pada 1 fold training lewat
[`resample_for_viz`](../../utils/imbalance_eval.py). Catatan: jumlah per kelas
tidak terpengaruh PCA (label utuh); PCA hanya relevan untuk scatter fitur.

## Catatan runtime
- **Oversampling** (atau custom dengan target besar) memperbesar training set →
  SVC `probability=True` lambat. Komentari model di `MODELS` bila perlu.
- Default backend GPU (`XGB_DEVICE='cuda'`, `CAT_TASK_TYPE='GPU'`) sesuai
  notebook 3.4.2 — ganti ke `'cpu'`/`'CPU'` kalau tak ada GPU.
- Butuh `imbalanced-learn` (`pip install imbalanced-learn`); sel pertama tiap
  notebook sudah meng-install-nya.
