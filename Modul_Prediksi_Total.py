import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

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

    # Grafik Prediksi vs Aktual
    df_plot = pd.melt(df_merge, id_vars='Tahun', value_vars=['Aktual', 'Prediksi'],
                      var_name='Tipe', value_name='Jumlah')
    fig = px.line(df_plot, x='Tahun', y='Jumlah', color='Tipe', markers=True,
                  title="Prediksi vs Aktual: Total Semua Layanan")

    fig.for_each_trace(
        lambda trace: trace.update(line=dict(color='gold')) if trace.name == 'Prediksi' else None
    )

    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

    # Tabel Prediksi Total dengan Nomor
    st.subheader("📊 Tabel Hasil Prediksi Total")
    df_tampil = df_merge.copy().reset_index(drop=True)
    df_tampil.index += 1  # Penomoran dimulai dari 1
    df_tampil = df_tampil[['Tahun', 'Aktual', 'Prediksi']]
    df_tampil.insert(0, 'Nomor', df_tampil.index)

    st.dataframe(df_tampil)

    return df_merge
