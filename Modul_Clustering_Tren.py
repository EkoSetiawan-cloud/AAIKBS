import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import plotly.express as px

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
    st.title("📈 Modul Klastering Berbasis Tren Statistik")

    st.markdown("""
    Modul ini mengelompokkan layanan berdasarkan **karakteristik tren statistik historis** seperti:
    - 📉 Kemiringan tren (`Slope`)
    - 📊 Rata-rata jumlah (`Mean`)
    - 📈 Fluktuasi antar tahun (`StdDev`, `Range`)
    - 🧬 Simetri data (`Skewness`)

    Dengan ini, kita bisa mengidentifikasi kelompok layanan yang naik, stabil, atau menurun.
    """)

    if df is None or df.empty:
        st.warning("⚠️ Dataset belum tersedia. Silakan input dan preprocessing terlebih dahulu.")
        return

    layanan_col = 'Layanan DJID'
    df_pivot = df.pivot_table(index=layanan_col, columns='Tahun', values='Jumlah', aggfunc='sum').fillna(0)
    df_fitur = compute_trend_features(df_pivot)
    fitur_scaled = StandardScaler().fit_transform(df_fitur)

    n_clusters = st.slider("🔢 Pilih jumlah klaster", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(fitur_scaled)

    df_result = df_fitur.copy()
    df_result['Cluster'] = labels
    df_result = df_result.reset_index()

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

    st.caption("📍 Visualisasi ini memetakan layanan dalam ruang 2 dimensi berdasarkan karakter tren statistik.")

    st.subheader("📋 Tabel Klaster dan Fitur Tren")
    st.dataframe(df_result.drop(columns=['PC1', 'PC2']), use_container_width=True)

    st.subheader("📝 Analisis Naratif per Klaster")
    narasi = ""
    df_naratif = []
    for clus in sorted(df_result['Cluster'].unique()):
        sub = df_result[df_result['Cluster'] == clus]
        mean_slope = sub['Slope'].mean()
        mean_std = sub['StdDev'].mean()

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

        if mean_std < 500:
            variasi = "stabil"
        elif mean_std < 1500:
            variasi = "moderat"
        elif mean_std < 5000:
            variasi = "fluktuatif"
        else:
            variasi = "sangat fluktuatif"

        narasi += f"- **Klaster {clus}** terdiri dari **{len(sub)} layanan**, tren **{tren}**, variasi **{variasi}**.\n"

        df_naratif.append({
            "Klaster": f"Klaster {clus}",
            "Jumlah Layanan": len(sub),
            "Tren Rata-rata": tren,
            "Variasi": variasi
        })

    st.markdown(narasi)

    st.subheader("📋 Tabel Ringkasan per Klaster")
    st.dataframe(pd.DataFrame(df_naratif), use_container_width=True, hide_index=True)

    st.subheader("📡 Radar Chart Karakteristik Klaster")
    df_radar = df_result.groupby('Cluster')[['Mean', 'StdDev', 'Range', 'Slope', 'Skewness']].mean().reset_index()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_radar.drop(columns=['Cluster']))
    df_scaled = pd.DataFrame(df_scaled, columns=df_radar.columns[1:])
    df_scaled['Cluster'] = df_radar['Cluster'].astype(str)

    categories = list(df_scaled.columns[:-1])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in df_scaled.iterrows():
        values = row[categories].tolist() + [row[categories[0]]]
        ax.plot(angles, values, label=f"Klaster {row['Cluster']}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Chart Fitur Statistik Tiap Klaster")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig_radar)

    st.caption("📡 Radar chart menunjukkan kekuatan dominan setiap fitur dalam masing-masing klaster.")

    st.subheader("🔥 Heatmap Fitur Tren per Layanan (Urut per Klaster)")
    df_heatmap = df_result.set_index('Layanan DJID').sort_values('Cluster')
    fig_hm, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_heatmap.drop(columns=['Cluster', 'PC1', 'PC2']), annot=True, fmt='.0f', cmap='YlGnBu', ax=ax2)
    ax2.set_title("Heatmap Fitur Tren")
    st.pyplot(fig_hm)

    st.caption("🔥 Heatmap ini membantu membandingkan nilai asli fitur tren antar layanan dalam klaster yang sama.")

    # === TREEMAP ===
    st.subheader("🌳 Treemap Komposisi Klaster")
    df_treemap = pd.DataFrame(df_naratif)

    fig_tree = px.treemap(
        df_treemap,
        path=[px.Constant("Semua Layanan"), 'Klaster', 'Tren Rata-rata'],
        values='Jumlah Layanan',
        color='Variasi',
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Treemap Komposisi Layanan Berdasarkan Klaster & Tren"
    )
    fig_tree.update_traces(root_color="lightgrey")
    fig_tree.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.caption("🌳 Treemap ini menampilkan proporsi jumlah layanan dalam satu root 'Semua Layanan', dibagi ke dalam klaster dan kategori tren rata-rata masing-masing.")

    return df_result
