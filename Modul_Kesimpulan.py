import streamlit as st
import pandas as pd
import numpy as np

def evaluasi_mape_kategori(mape):
    if mape <= 10:
        return "Sangat Akurat (Highly Accurate)"
    elif mape <= 20:
        return "Akurat (Good Forecast)"
    elif mape <= 50:
        return "Cukup Akurat (Reasonable Forecast)"
    else:
        return "Tidak Akurat (Inaccurate Forecast)"

def layanan_aktif(df):
    layanan_unik = df['Layanan'].dropna().unique()
    if len(layanan_unik) == 1:
        return layanan_unik[0]
    return "Semua Layanan"

def modul_kesimpulan(df_eval_total):
    st.title("🧾 Kesimpulan Evaluasi Model Prediksi")

    if df_eval_total is None or df_eval_total.empty:
        st.warning("⚠️ Data evaluasi belum tersedia. Jalankan Modul Evaluasi.")
        return

    df_historis = df_eval_total[df_eval_total['Aktual'].notna()].copy()
    nama_layanan = layanan_aktif(df_eval_total)

    # === 📈 Skor Evaluasi Global Historis
    st.subheader("📈 Skor Evaluasi Global Historis")
    global_mae = df_historis['MAE'].mean()
    global_rmse = np.sqrt(df_historis['RMSE'].mean())
    global_mape = df_historis['MAPE (%)'].mean()
    global_kategori = evaluasi_mape_kategori(global_mape)

    df_global = pd.DataFrame([{
        "Layanan": nama_layanan,
        "MAE": round(global_mae, 2),
        "RMSE": round(global_rmse, 2),
        "MAPE (%)": round(global_mape, 2),
        "Validasi Akurasi": global_kategori
    }])
    st.dataframe(df_global, use_container_width=True, hide_index=True)

    # === ⚖️ Skor Evaluasi Berbasis Bobot
    st.subheader("⚖️ Skor Evaluasi Global Berbasis Bobot (Weighted by Aktual)")
    df_historis['Bobot'] = df_historis['Aktual'] / df_historis['Aktual'].sum()

    mae_w = np.average(df_historis['MAE'], weights=df_historis['Bobot'])
    rmse_w = np.sqrt(np.average(df_historis['RMSE'], weights=df_historis['Bobot']))
    mape_w = np.average(df_historis['MAPE (%)'], weights=df_historis['Bobot'])
    kategori_w = evaluasi_mape_kategori(mape_w)

    df_weighted = pd.DataFrame([{
        "Layanan": nama_layanan,
        "MAE (Weighted)": round(mae_w, 2),
        "RMSE (Weighted)": round(rmse_w, 2),
        "MAPE (%) (Weighted)": round(mape_w, 2),
        "Validasi Akurasi": kategori_w
    }])
    st.dataframe(df_weighted, use_container_width=True, hide_index=True)

    # === Narasi Otomatis
    st.subheader("📝 Kesimpulan")

    kesimpulan = f"""
Evaluasi terhadap layanan **{nama_layanan}** menghasilkan:

- MAE rata-rata: **{global_mae:,.2f}**
- RMSE rata-rata: **{global_rmse:,.2f}**
- MAPE rata-rata: **{global_mape:.2f}%** → **{global_kategori}**

Jika dihitung berbobot terhadap jumlah aktual layanan per tahun:

- MAE berbobot: **{mae_w:,.2f}**
- RMSE berbobot: **{rmse_w:,.2f}**
- MAPE berbobot: **{mape_w:.2f}%** → **{kategori_w}**

Kesimpulan: Model memiliki performa prediksi yang sangat baik dan layak digunakan sebagai dasar evaluasi serta perencanaan untuk layanan ini.
"""
    st.markdown(kesimpulan)
