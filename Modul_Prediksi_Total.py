# Modul_Prediksi_Total.py

import streamlit as st
import pandas as pd
from prophet import Prophet
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

def modul_prediksi_total(df):
    st.title("🔮 Prediksi Total Jumlah Layanan DJID (Semua Layanan Digabung)")

    if df is None:
        st.warning("⚠️ Data belum tersedia.")
        return

    # Agregasi total layanan per tahun
    df_total = df.groupby('Tahun')['Jumlah'].sum().reset_index()
    df_total.columns = ['Tahun', 'Aktual']
    df_total['ds'] = pd.to_datetime(df_total['Tahun'], format='%Y')
    df_total['y'] = df_total['Aktual']

    # Fit model Prophet
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(df_total[['ds', 'y']])

    # Buat prediksi 3 tahun ke depan
    future = model.make_future_dataframe(periods=3, freq='Y')
    forecast = model.predict(future)

    # Gabungkan prediksi dengan aktual
    pred_df = forecast[['ds', 'yhat']].copy()
    pred_df['Tahun'] = pred_df['ds'].dt.year
    pred_df.rename(columns={'yhat': 'Prediksi'}, inplace=True)
    df_merge = pd.merge(pred_df[['Tahun', 'Prediksi']], df_total[['Tahun', 'Aktual']], on='Tahun', how='left')

    # Tampilkan grafik Prediksi vs Aktual
    df_plot = pd.melt(df_merge, id_vars='Tahun', value_vars=['Aktual', 'Prediksi'],
                      var_name='Tipe', value_name='Jumlah')
    fig = px.line(df_plot, x='Tahun', y='Jumlah', color='Tipe', markers=True,
              title="Prediksi vs Aktual: Total Semua Layanan")

    # Ubah warna garis prediksi menjadi gold
    fig.for_each_trace(
        lambda trace: trace.update(line=dict(color='gold')) if trace.name == 'Prediksi' else None
    )

    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Tabel hasil prediksi
    st.subheader("📊 Tabel Hasil Prediksi Total")
    st.dataframe(df_merge)

    # ✅ Tabel Evaluasi Akurasi per Tahun
    st.subheader("✅ Evaluasi Akurasi Total Model Prophet per Tahun")

    df_eval = df_merge.dropna()  # hanya tahun yang ada aktual dan prediksi
    df_eval['MAE'] = abs(df_eval['Aktual'] - df_eval['Prediksi'])
    df_eval['RMSE'] = (df_eval['Aktual'] - df_eval['Prediksi'])**2
    df_eval['MAPE (%)'] = (abs(df_eval['Aktual'] - df_eval['Prediksi']) / df_eval['Aktual']) * 100
    df_eval['MAPE (%)'] = df_eval['MAPE (%)'].round(2)
    df_eval['Validasi Akurasi'] = df_eval['MAPE (%)'].apply(evaluasi_mape_kategori)

    st.dataframe(df_eval[['Tahun', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']])

    return df_merge
