import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def compute_trend_features(df_pivot):
    tahun = np.array(df_pivot.columns, dtype=int)
    fitur = pd.DataFrame(index=df_pivot.index)

    fitur['Mean'] = df_pivot.mean(axis=1)
    fitur['StdDev'] = df_pivot.std(axis=1)
    fitur['Range'] = df_pivot.max(axis=1) - df_pivot.min(axis=1)
    fitur['Slope'] = df_pivot.apply(lambda row: np.polyfit(tahun, row.values, 1)[0], axis=1)
    fitur['Skewness'] = df_pivot.skew(axis=1)

    return fitur

def modul_clustering_tren(df):
    st.title("📈 Modul Klastering Berbasis Tren")

    if df is None or df.empty:
        st.warning("⚠️ Dataset belum tersedia. Silakan input dan preprocessing terlebih dahulu.")
        return

    layanan_col = 'Layanan DJID'
    df_pivot = df.pivot_table(index=layanan_col, columns='Tahun', values='Jumlah', aggfunc='sum').fillna(0)

    # === Ekstraksi fitur tren statistik
    df_fitur = compute_trend_features(df_pivot)
    fitur_scaled = StandardScaler().fit_transform(df_fitur)

    # === Klastering
    n_clusters = st.slider("Pilih jumlah klaster", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(fitur_scaled)

    df_result = df_fitur.copy()
    df_result['Cluster'] = labels
    df_result = df_result.reset_index()

    # === Visualisasi PCA 2D
    st.subheader("📊 Visualisasi Klaster (PCA 2D)")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(fitur_scaled)
    df_result['PC1'] = pca_result[:, 0]
    df_result['PC2'] = pca_result[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_result, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100)
    for i in range(len(df_result)):
        ax.text(df_result['PC1'][i]+0.02, df_result['PC2'][i], df_result[layanan_col][i], fontsize=9)
    plt.title("Distribusi Klaster berdasarkan Tren Statistik (PCA)")
    st.pyplot(fig)

    # === Tabel hasil
    st.subheader("📋 Tabel Klaster dan Fitur Tren")
    st.dataframe(df_result.drop(columns=['PC1', 'PC2']), use_container_width=True)

    # === Narasi Otomatis Granular
    st.subheader("📝 Klaster Berdasarkan Tren")

    narasi = ""
    for clus in sorted(df_result['Cluster'].unique()):
        sub = df_result[df_result['Cluster'] == clus]
        mean_slope = sub['Slope'].mean()
        mean_std = sub['StdDev'].mean()

        # Klasifikasi slope
        if mean_slope > 1000:
            tren = "meningkat sangat tajam"
        elif mean_slope > 100:
            tren = "meningkat signifikan"
        elif mean_slope > 10:
            tren = "meningkat pelan"
        elif mean_slope > -10:
            tren = "relatif stabil"
        elif mean_slope > -100:
            tren = "menurun pelan"
        elif mean_slope > -1000:
            tren = "menurun signifikan"
        else:
            tren = "menurun sangat tajam"

        # Klasifikasi stddev
        if mean_std < 500:
            var = "stabil"
        elif mean_std < 1500:
            var = "moderat"
        elif mean_std < 5000:
            var = "fluktuatif"
        else:
            var = "sangat fluktuatif"

        narasi += f"- **Klaster {clus}** berisi **{len(sub)} layanan**, dengan tren **{tren}** dan variasi **{var}**.\n"

    st.markdown(narasi)

    # === 📡 Radar Chart
    st.subheader("📡 Radar Chart Karakteristik Tiap Klaster")
    df_radar = df_result.groupby('Cluster')[['Mean', 'StdDev', 'Range', 'Slope', 'Skewness']].mean().reset_index()

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_radar.drop(columns=['Cluster']))
    df_scaled = pd.DataFrame(df_scaled, columns=df_radar.columns[1:])
    df_scaled['Cluster'] = df_radar['Cluster'].astype(str)

    categories = list(df_scaled.columns[:-1])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in df_scaled.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Klaster {row['Cluster']}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Chart Rata-Rata Fitur Tiap Klaster")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig_radar)

    st.caption("""
📡 **Radar Chart** ini menunjukkan karakteristik rata-rata dari masing-masing klaster.
Setiap sumbu merepresentasikan fitur statistik tren layanan, seperti:
- `Mean`: Rata-rata jumlah layanan
- `StdDev`: Fluktuasi antar tahun
- `Range`: Rentang perubahan
- `Slope`: Arah tren (naik/turun)
- `Skewness`: Simetri distribusi tren

Garis yang lebih besar di satu area menandakan kekuatan dominan fitur tersebut dalam klaster.
""")

    # === 🔥 Heatmap
    st.subheader("🔥 Heatmap Fitur Tren per Layanan (Urut per Klaster)")
    df_heatmap = df_result.set_index('Layanan DJID').sort_values('Cluster')
    fig_hm, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_heatmap.drop(columns=['Cluster', 'PC1', 'PC2']), annot=True, fmt='.0f', cmap='YlGnBu', ax=ax2)
    ax2.set_title("Heatmap Fitur Tren")
    st.pyplot(fig_hm)

    st.caption("""
🔥 **Heatmap** ini menampilkan nilai asli fitur tren untuk setiap layanan, diurutkan berdasarkan klasternya.
Semakin gelap warnanya, semakin tinggi nilainya. 
Gunakan ini untuk melihat:
- Layanan mana yang paling fluktuatif
- Siapa yang paling tajam tren naik/turunnya
- Perbandingan visual antar layanan dalam satu klaster
""")

    return df_result
