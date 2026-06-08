# Tahap 7 — Korelasi Fitur dan Target

Notebook: [`7.feature_correlation`](../notebooks/7.feature_correlation.ipynb)

## Tujuan
Memetakan hubungan antara fitur sensor/sensorik dengan target mutu teh hijau, sebagai dasar
seleksi fitur dan pemahaman data.

## Implementasi
- Target dapat diganti via variabel `TARGET` (`"Kategori"` atau `"Standar Kualitas"`).
- **Encoding**:
  - Target kategorikal → ordinal urut alfabet (`Kategori` A–E → 0…4 = jenjang kualitas;
    `Standar Kualitas` Baik/Cacat Mutu → 0/1).
  - `Chop_ID` & `Sampling_ID` → kode integer (nominal).
- **Metode korelasi: Spearman** — dipilih karena target ordinal dan banyak hubungan
  non-linear/monoton.
- Heatmap `seaborn` (annot, `coolwarm`, `vmin/vmax = -1/1`) atas seluruh fitur:
  12 sensor e-nose + 5 sensorik (`Aroma, Taste, Color, Appearance, Dreg`) + `Chop_ID`/`Sampling_ID`.

## Peringatan penting (terkait kebocoran data)
> ⚠️ `Chop_ID`/`Sampling_ID` hanya **kode urutan ID**. Korelasinya bisa **menyesatkan** dan
> terkait isu **kebocoran data ber-grup** — **jangan dijadikan fitur model**.

Ini konsisten dengan keputusan tahap 3.4.1 yang memakai `Sampling_ID` **hanya sebagai `groups`**
untuk **StratifiedGroupKFold**, bukan sebagai fitur prediktif. Lihat [README.md](README.md) untuk
penjelasan SKF vs SGKF dan kebocoran ber-grup.

## Catatan Imbalance & CV
- Notebook ini bersifat analitik (EDA korelasi), tidak melakukan training/CV.
- Membantu memahami fitur mana yang paling informatif terhadap kelas yang timpang, dan menegaskan
  agar kolom identitas grup tidak mencemari model.
