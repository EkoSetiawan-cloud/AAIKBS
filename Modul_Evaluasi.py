# Modul_Evaluasi.py

import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.express as px

def evaluasi_mape_kategori(mape):
    if mape <= 10:
        return "Sangat Akurat (Highly Accurate)"
    elif mape <= 20:
        return "Akurat (Good Forecast)"
    elif mape <= 50:
        return "Cukup Akurat (Reasonable Forecast)"
    else:
        return "Tidak Akurat (Inaccurate Forecast)"

def modul_evaluasi(df_aktual, df_prediksi):
    st.title("📉 Modul Evaluasi Prediksi")

    st.markdown(
        """
        Modul ini mengevaluasi hasil prediksi terhadap data aktual menggunakan metrik:
        - **MAE (Mean Absolute Error)**
        - **RMSE (Root Mean Squared Error)**
        - **MAPE (Mean Absolute Percentage Error)**  
        
        Ditambah klasifikasi akurasi berdasarkan nilai MAPE:
        - ≤ 10%: Sangat Akurat
        - ≤ 20%: Akurat
        - ≤ 50%: Cukup Akurat
        - > 50%: Tidak Akurat
        """
    )

    if df_aktual is None or df_prediksi is None:
        st.warning("⚠️ Data aktual atau prediksi tidak tersedia.")
        return

    required_cols = {'Tahun', 'Layanan', 'Jumlah'}
    if not required_cols.issubset(set(df_aktual.columns)):
        st.error("❌ Dataset aktual harus memiliki kolom: Tahun, Layanan, Jumlah")
        return
    if 'Prediksi' not in df_prediksi.columns:
        st.error("❌ Dataset prediksi harus memiliki kolom 'Prediksi'")
        return

    df_aktual_renamed = df_aktual.rename(columns={'Jumlah': 'Aktual'})
    df_prediksi_renamed = df_prediksi[['Tahun', 'Layanan', 'Prediksi']]
    df_merged = pd.merge(df_aktual_renamed, df_prediksi_renamed, on=['Tahun', 'Layanan'], how='inner')

    if df_merged.empty:
        st.error("❌ Tidak ditemukan data yang cocok antara aktual dan prediksi.")
        return

    # Evaluasi per Layanan
    eval_list = []
    for layanan in df_merged['Layanan'].unique():
        data = df_merged[df_merged['Layanan'] == layanan]
        y_true = data['Aktual'].values
        y_pred = data['Prediksi'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        eval_list.append({
            'Layanan': layanan,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': round(mape, 2),
            'Kategori Akurasi': evaluasi_mape_kategori(mape)
        })

    df_evaluasi = pd.DataFrame(eval_list).sort_values('MAPE (%)')

    # Tambahkan kolom validasi ke dataframe evaluasi
    df_evaluasi['Validasi Akurasi'] = df_evaluasi['MAPE (%)'].apply(evaluasi_mape_kategori)

    st.subheader("📋 Tabel Evaluasi per Layanan")
    st.dataframe(df_evaluasi)

    # Grafik Ranking
    st.subheader("🏅 Grafik Ranking Akurasi Model per Layanan")
    fig_rank = px.bar(df_evaluasi, x='Layanan', y='MAPE (%)', color='Kategori Akurasi',
                      text='MAPE (%)', title="Ranking MAPE (%)", height=450)
    fig_rank.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_rank.update_layout(xaxis_title='Layanan', yaxis_title='MAPE (%)', uniformtext_minsize=8)
    st.plotly_chart(fig_rank, use_container_width=True)

    # Grafik Line per Layanan (optional)
    st.subheader("📈 Grafik Aktual vs Prediksi per Layanan")
    layanan_terpilih = st.selectbox("Pilih Layanan", df_merged['Layanan'].unique())
    df_layanan = df_merged[df_merged['Layanan'] == layanan_terpilih]

    df_plot = pd.melt(df_layanan, id_vars='Tahun', value_vars=['Aktual', 'Prediksi'],
                      var_name='Tipe', value_name='Jumlah')
    fig_line = px.line(df_plot, x='Tahun', y='Jumlah', color='Tipe', markers=True,
                       title=f"Perbandingan Aktual vs Prediksi: {layanan_terpilih}")
    st.plotly_chart(fig_line, use_container_width=True)

    return df_evaluasi, df_merged
