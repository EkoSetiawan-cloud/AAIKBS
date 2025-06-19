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

    df_evaluasi = df_merge[df_merge['Aktual'].notna()].drop_duplicates(subset='Tahun')
    df_evaluasi['MAE'] = abs(df_evaluasi['Aktual'] - df_evaluasi['Prediksi'])
    df_evaluasi['RMSE'] = (df_evaluasi['Aktual'] - df_evaluasi['Prediksi'])**2
    df_evaluasi['MAPE (%)'] = (abs(df_evaluasi['Aktual'] - df_evaluasi['Prediksi']) / df_evaluasi['Aktual']) * 100
    df_evaluasi['MAPE (%)'] = df_evaluasi['MAPE (%)'].round(2)
    df_evaluasi['Validasi Akurasi'] = df_evaluasi['MAPE (%)'].apply(evaluasi_mape_kategori)

    st.dataframe(df_evaluasi[['Tahun', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']])

    # ✅ Tabel Prediksi Masa Depan (Estimasi MAPE vs tahun terakhir)
    st.subheader("🔮 Estimasi Evaluasi Prediksi 3 Tahun ke Depan")
    df_future = df_merge[df_merge['Aktual'].isna()].copy()

    last_actual = df_total.iloc[-1]['Aktual']
    df_future['Aktual Estimasi'] = last_actual
    df_future['MAE'] = abs(df_future['Aktual Estimasi'] - df_future['Prediksi'])
    df_future['RMSE'] = (df_future['Aktual Estimasi'] - df_future['Prediksi'])**2
    df_future['MAPE (%)'] = (abs(df_future['Aktual Estimasi'] - df_future['Prediksi']) / df_future['Aktual Estimasi']) * 100
    df_future['MAPE (%)'] = df_future['MAPE (%)'].round(2)
    df_future['Validasi Akurasi'] = df_future['MAPE (%)'].apply(evaluasi_mape_kategori)

    st.dataframe(df_future[['Tahun', 'Prediksi', 'Aktual Estimasi', 'MAE', 'RMSE', 'MAPE (%)', 'Validasi Akurasi']])

    return df_merge
