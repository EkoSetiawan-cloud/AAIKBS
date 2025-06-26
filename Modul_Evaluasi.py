import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluasi_mape_kategori(mape):
    if mape <= 10:
        return "Sangat Akurat (Highly Accurate)"
    elif mape <= 20:
        return "Akurat (Good Forecast)"
    elif mape <= 50:
        return "Cukup Akurat (Reasonable Forecast)"
    else:
        return "Tidak Akurat (Inaccurate Forecast)"

def modul_evaluasi(df_merge):
    st.title("📊 Modul Evaluasi Model Prediksi")

    st.markdown("""
    Modul ini mengevaluasi performa model prediksi menggunakan metrik evaluasi umum:
    - **MAE** (Mean Absolute Error)
    - **RMSE** (Root Mean Squared Error)
    - **MAPE (%)** (Mean Absolute Percentage Error)

    Evaluasi dilakukan berdasarkan data historis aktual dan prediksi, serta estimasi performa masa depan.
    """)

    if df_merge is None or df_merge.empty:
        st.warning("⚠️ Data belum tersedia dari modul prediksi.")
        return

    layanan_terpilih = st.selectbox("📌 Pilih Layanan untuk Evaluasi", sorted(df_merge['Layanan'].unique()))
    df_layanan = df_merge[df_merge['Layanan'] == layanan_terpilih].copy()

    if df_layanan.empty:
        st.error("❌ Tidak ada data untuk layanan yang dipilih.")
        return

    st.subheader("📋 Evaluasi Error pada Data Historis")
    df_error = df_layanan[df_layanan['Aktual'].notna() & (df_layanan['Aktual'] != 0)].copy()
    df_error = df_error.drop_duplicates(subset='Tahun', keep='first')

    if df_error.empty:
        st.warning("⚠️ Tidak ada data aktual yang valid untuk evaluasi error.")
    else:
        df_error['Error Absolute'] = (df_error['Aktual'] - df_error['Prediksi']).abs()
        df_error['Error Persentase (%)'] = ((df_error['Error Absolute'] / df_error['Aktual']) * 100).round(2)
        df_error['MAE'] = df_error['Error Absolute']
        df_error['RMSE'] = (df_error['Aktual'] - df_error['Prediksi']) ** 2
        df_error['MAPE (%)'] = df_error['Error Persentase (%)']
        df_error['Validasi Akurasi'] = df_error['MAPE (%)'].apply(evaluasi_mape_kategori)

        st.dataframe(df_error[['Layanan', 'Tahun', 'Aktual', 'Prediksi', 'Error Absolute', 'Error Persentase (%)']],
                     use_container_width=True, hide_index=True)

    st.subheader("📈 Tabel Evaluasi Performa per Tahun")
    df_eval_summary = df_error[['Layanan', 'Tahun', 'Aktual', 'Prediksi', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']].copy()
    st.dataframe(df_eval_summary, use_container_width=True, hide_index=True)

    st.caption("📌 Tabel ini menampilkan performa prediksi berdasarkan data historis aktual.")

    st.subheader("🔮 Estimasi Evaluasi 2 Tahun ke Depan")
    df_future = df_layanan[~df_layanan['Tahun'].isin(df_error['Tahun'])].copy()

    baseline_aktual = df_error['Aktual'].iloc[-1] if not df_error.empty else 1
    df_future['Aktual (Estimasi)'] = baseline_aktual
    df_future['MAE'] = (df_future['Aktual (Estimasi)'] - df_future['Prediksi']).abs()
    df_future['RMSE'] = (df_future['Aktual (Estimasi)'] - df_future['Prediksi']) ** 2
    df_future['MAPE (%)'] = ((df_future['MAE'] / df_future['Aktual (Estimasi)']) * 100).round(2)
    df_future['Validasi Akurasi'] = df_future['MAPE (%)'].apply(evaluasi_mape_kategori)
    df_future['Layanan'] = layanan_terpilih

    st.dataframe(df_future[['Layanan', 'Tahun', 'Prediksi', 'Aktual (Estimasi)', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']],
                 use_container_width=True, hide_index=True)

    st.caption("📌 Estimasi ini digunakan untuk melihat performa model pada periode mendatang menggunakan baseline aktual terakhir.")

    st.subheader("📊 Ringkasan Evaluasi Global Historis")
    if not df_error.empty:
        global_mae = df_error['MAE'].mean()
        global_rmse = np.sqrt(df_error['RMSE'].mean())
        global_mape = df_error['MAPE (%)'].mean()

        df_global = pd.DataFrame([{
            "Layanan": layanan_terpilih,
            "MAE": round(global_mae, 2),
            "RMSE": round(global_rmse, 2),
            "MAPE (%)": round(global_mape, 2),
            "Validasi Akurasi": evaluasi_mape_kategori(global_mape)
        }])

        st.dataframe(df_global, use_container_width=True, hide_index=True)

        st.caption("📌 Ringkasan ini menunjukkan akurasi rata-rata model pada data historis.")

    return df_eval_summary
