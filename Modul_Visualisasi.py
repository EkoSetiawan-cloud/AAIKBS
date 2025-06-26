import streamlit as st
import pandas as pd
import plotly.express as px

def modul_visualisasi(df_prediksi, df_evaluasi):
    st.title("📊 Modul Visualisasi Prediksi dan Evaluasi")

    st.markdown(
        """
        Modul ini menyajikan **visualisasi interaktif** dari hasil prediksi dan evaluasi model Prophet
        yang telah dijalankan pada layanan DJID.
        """
    )

    if df_prediksi is None or df_evaluasi is None:
        st.warning("⚠️ Data prediksi atau evaluasi tidak tersedia. Jalankan Modul Prediksi terlebih dahulu.")
        return

    # --- Visualisasi Hasil Prediksi ---
    st.subheader("📈 Grafik Prediksi Jumlah Layanan per Tahun")
    layanan_terpilih = st.multiselect("Pilih Layanan", 
                                      options=df_prediksi['Layanan'].unique(), 
                                      default=df_prediksi['Layanan'].unique())

    df_filter = df_prediksi[df_prediksi['Layanan'].isin(layanan_terpilih)]
    fig = px.line(df_filter, x='Tahun', y='Prediksi', color='Layanan',
                  title="Prediksi Jumlah per Layanan", markers=True)
    fig.update_layout(template='simple_white')
    st.plotly_chart(fig, use_container_width=True)

    # --- Visualisasi Tabel Evaluasi ---
    st.subheader("📋 Tabel Evaluasi MAPE per Layanan")
    st.dataframe(df_evaluasi.sort_values('MAPE (%)'))

    # --- Visualisasi Ranking MAPE ---
    st.subheader("🏅 Grafik Ranking Akurasi Model (Berdasarkan MAPE)")
    df_rank = df_evaluasi.copy()
    df_rank['Ranking'] = df_rank['MAPE (%)'].rank(method='min')
    fig2 = px.bar(df_rank.sort_values('MAPE (%)'),
                  x='Layanan', y='MAPE (%)', color='Validasi Akurasi',
                  text='MAPE (%)',
                  title="Ranking Akurasi Model per Layanan")
    fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig2, use_container_width=True)
