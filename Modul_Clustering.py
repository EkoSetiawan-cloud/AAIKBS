# Modul_Clustering.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def modul_clustering(df):
    st.title("🔗 Modul Klastering: KMeans + Heatmap")

    st.markdown(
        """
        Modul ini menggunakan algoritma **KMeans Clustering** untuk mengelompokkan layanan
        berdasarkan tren tahunan, dan menampilkan hasilnya dalam bentuk **heatmap**.
        """
    )

    if df is None:
        st.warning("⚠️ Silakan unggah dan preprocessing dataset terlebih dahulu.")
        return

    # Tetapkan langsung nama kolom identitas layanan
    layanan_col = 'Layanan DJID'

    # Pivot data: index = layanan, kolom = tahun, nilai = jumlah
    df_pivot = df.pivot_table(index=layanan_col, columns='Tahun', values='Jumlah', aggfunc='sum')
    df_pivot = df_pivot.fillna(0)

    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pivot)

    # Klastering
    n_clusters = st.slider("Pilih jumlah klaster", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Gabungkan hasil
    df_clustered = df_pivot.copy()
    df_clustered['Cluster'] = cluster_labels
    df_clustered_sorted = df_clustered.sort_values(by='Cluster')

    # Tampilkan tabel hasil klaster
    st.subheader("📋 Tabel Hasil Klasterisasi")
    st.dataframe(df_clustered_sorted)

    # Visualisasi heatmap
    st.subheader("🔥 Heatmap Pola Tren Layanan per Cluster")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df_clustered_sorted.drop(columns='Cluster'),
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        ax=ax,
        yticklabels=[
            f"{idx} (Cluster {row['Cluster']})"
            for idx, row in df_clustered_sorted.iterrows()
        ]
    )
    ax.set_title("Klasterisasi Layanan Berdasarkan Pola Tren (2019–2024)")
    st.pyplot(fig)

    return df_clustered_sorted.reset_index()
