import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.express as px

# Fungsi validasi berdasarkan MAPE
def evaluasi_mape_kategori(mape):
    if mape <= 10:
        return "Sangat Akurat (Highly Accurate)"
    elif mape <= 20:
        return "Akurat (Good Forecast)"
    elif mape <= 50:
        return "Cukup Akurat (Reasonable Forecast)"
    else:
        return "Tidak Akurat (Inaccurate Forecast)"

def modul_prediksi(df):
    st.title("🔮 Modul Prediksi: Facebook Prophet")

    st.markdown(
        """
        Modul ini memprediksi jumlah layanan berdasarkan data historis dari tahun pertama hingga tahun terakhir,
        serta menambahkan prediksi **3 tahun ke depan**.
        """
    )

    if df is None:
        st.warning("⚠️ Silakan unggah dan preprocessing dataset terlebih dahulu.")
        return None, None

    layanan_col = 'Layanan DJID'
    layanan_list = df[layanan_col].unique()
    forecast_list = []
    eval_rows = []
    gabungan_list = []

    for layanan in layanan_list:
        data_layanan = df[df[layanan_col] == layanan][['Tahun', 'Jumlah']].copy()
        data_layanan.columns = ['Tahun', 'Aktual']
        prophet_data = data_layanan.copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')

        # Fit model Prophet
        model = Prophet(yearly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_data)

        # Prediksi 3 tahun ke depan
        future = model.make_future_dataframe(periods=3, freq='Y')
        forecast = model.predict(future)

        pred = forecast[['ds', 'yhat']].copy()
        pred['Tahun'] = pred['ds'].dt.year
        pred['Layanan'] = layanan
        pred.rename(columns={'yhat': 'Prediksi'}, inplace=True)

        # Gabungkan prediksi dengan aktual
        gabung = pd.merge(pred[['Tahun', 'Layanan', 'Prediksi']], data_layanan, on=['Tahun'], how='left')
        forecast_list.append(pred[['Tahun', 'Layanan', 'Prediksi']])
        gabungan_list.append(gabung)

        # Evaluasi (untuk disimpan meski tidak ditampilkan)
        y_true = prophet_data['y'].values
        y_pred = model.predict(prophet_data)['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        eval_rows.append({
            'Layanan': layanan,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': round(mape, 2),
            'Validasi Akurasi': evaluasi_mape_kategori(mape)
        })

    df_prediksi = pd.concat(gabungan_list, ignore_index=True)
    df_evaluasi = pd.DataFrame(eval_rows)

    # Tabel hasil prediksi
    st.subheader("📊 Tabel Hasil Prediksi (termasuk aktual & prediksi)")
    st.dataframe(df_prediksi)

    # Grafik Prediksi vs Aktual
    st.subheader("📈 Grafik Prediksi dan Aktual per Layanan")
    layanan_terpilih = st.selectbox("Pilih Layanan", df_prediksi['Layanan'].unique())

    df_plot = df_prediksi[df_prediksi['Layanan'] == layanan_terpilih].copy()
    df_long = pd.melt(df_plot, id_vars='Tahun', value_vars=['Aktual', 'Prediksi'],
                      var_name='Tipe', value_name='Jumlah')

    fig = px.line(df_long, x='Tahun', y='Jumlah', color='Tipe', markers=True,
                  title=f"Prediksi vs Aktual: {layanan_terpilih}")

    fig.for_each_trace(
        lambda trace: trace.update(line=dict(color='gold')) if trace.name == 'Prediksi' else None
    )

    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Tahun', yaxis_title='Jumlah Layanan')
    st.plotly_chart(fig, use_container_width=True)

    return df_prediksi, df_evaluasi
