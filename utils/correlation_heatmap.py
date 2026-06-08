"""Heatmap korelasi fitur terhadap target (Kategori) untuk Dataset Teh Hijau.

Fitur yang dipakai mengikuti permintaan: sensor gas, lingkungan, ID, dan
penilaian sensorik. Kolom kategorikal (Chop_ID, Sampling_ID, Kategori)
di-encode menjadi numerik sebelum menghitung korelasi Spearman.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "dataset", "Dataset_TehHijau.csv")
OUT_PATH = os.path.join(ROOT, "dataset", "korelasi_fitur_target.png")

FEATURES = [
    "MQ3", "TGS822", "TGS2602", "MQ5", "MQ138", "TGS2620", "TGS813",
    "TGS2600", "TGS2611", "TGS2603", "Humidity", "Celsius",
    "Chop_ID", "Sampling_ID", "Aroma", "Taste", "Color", "Appearance", "Dreg",
]
TARGET = "Kategori"


def main():
    df = pd.read_csv(DATA_PATH)
    data = df[FEATURES + [TARGET]].copy()

    # Encode kolom kategorikal menjadi numerik.
    # Kategori A..E bersifat ordinal (jenjang kualitas) -> A=0 .. E=4.
    cat_order = sorted(data[TARGET].dropna().unique())
    data[TARGET] = data[TARGET].map({c: i for i, c in enumerate(cat_order)})

    # Chop_ID & Sampling_ID adalah label nominal -> kode integer.
    for col in ["Chop_ID", "Sampling_ID"]:
        data[col] = data[col].astype("category").cat.codes

    # Spearman: robust untuk hubungan monoton & data ordinal.
    corr = data.corr(method="spearman")

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Korelasi Spearman"},
        annot_kws={"size": 7},
    )
    plt.title("Korelasi Fitur dan Target (Kategori) - Dataset Teh Hijau", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Heatmap disimpan ke: {OUT_PATH}")

    # Tampilkan ringkasan korelasi terhadap target.
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\nKorelasi fitur terhadap Kategori (urut |nilai| terbesar):")
    print(target_corr.to_string())


if __name__ == "__main__":
    main()
