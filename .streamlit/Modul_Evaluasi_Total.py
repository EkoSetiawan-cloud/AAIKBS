# Modul_Evaluasi_Total.py

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

def modul_evaluasi_total(df_merge):
    st.title("📊 Modul Evaluasi Total: Performa Model Prophet")

    if df_merge is None or df_merge.empty:
        st.warning("⚠️ Data belum tersedia dari modul prediksi total.")
        return

    # === 1. Tabel Evaluasi Error ===
    st.subheader("📋 Tabel Evaluasi Error per Tahun")
    df_error = df_merge[df_merge['Aktual'].notna()].copy()
    df_error['Error Absolute'] = (df_error['Aktual'] - df_error['Prediksi']).abs()
    df_error['Error Persentase (%)'] = ((df_error['Error Absolute'] / df_error['Aktual']) * 100).round(2)
    st.dataframe(df_error[['Tahun', 'Aktual', 'Prediksi', 'Error Absolute', 'Error Persentase (%)']])

    # === 2. Evaluasi Performa Model ===
    st.subheader("📈 Tabel Evaluasi Performa Model per Tahun")
    df_eval = df_merge[df_merge['Aktual'].notna()].copy()
    df_eval['MAE'] = (df_eval['Aktual'] - df_eval['Prediksi']).abs()
    df_eval['RMSE'] = (df_eval['Aktual'] - df_eval['Prediksi'])**2
    df_eval['MAPE (%)'] = ((df_eval['MAE'] / df_eval['Aktual']) * 100).round(2)
    df_eval['Validasi Akurasi'] = df_eval['MAPE (%)'].apply(evaluasi_mape_kategori)

    df_eval_summary = df_eval[['Tahun', 'Aktual', 'Prediksi', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']]
    st.dataframe(df_eval_summary)

    # === 3. Prediksi Masa Depan (Tanpa Aktual) ===
    st.subheader("🔮 Prediksi 2 Tahun ke Depan (Estimasi Evaluasi)")
    baseline_aktual = df_eval['Aktual'].iloc[-1]  # gunakan nilai aktual terakhir sebagai pembanding

    df_future = df_merge[df_merge['Aktual'].isna()].copy()
    df_future['Aktual (Estimasi)'] = baseline_aktual
    df_future['MAE'] = (df_future['Aktual (Estimasi)'] - df_future['Prediksi']).abs()
    df_future['RMSE'] = (df_future['Aktual (Estimasi)'] - df_future['Prediksi'])**2
    df_future['MAPE (%)'] = ((df_future['MAE'] / df_future['Aktual (Estimasi)']) * 100).round(2)
    df_future['Validasi Akurasi'] = df_future['MAPE (%)'].apply(evaluasi_mape_kategori)

    st.dataframe(df_future[['Tahun', 'Prediksi', 'Aktual (Estimasi)', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']])

    # === 4. Ringkasan Global ===
    st.subheader("📊 Ringkasan Skor Evaluasi Global")
    global_mae = df_eval['MAE'].mean()
    global_rmse = np.sqrt(df_eval['RMSE'].mean())
    global_mape = df_eval['MAPE (%)'].mean()

    df_global = pd.DataFrame([{
        "Model": "Prophet (Total)",
        "MAE": round(global_mae, 2),
        "RMSE": round(global_rmse, 2),
        "MAPE (%)": round(global_mape, 2),
        "Validasi Akurasi": evaluasi_mape_kategori(global_mape)
    }])
    st.dataframe(df_global)

    # ✅ Kembalikan evaluasi per tahun ke modul utama
    return df_eval_summary
