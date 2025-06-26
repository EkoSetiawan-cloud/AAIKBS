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
    st.title("🧾 Modul Kesimpulan Evaluasi Prediktif")

    st.markdown("""
    Modul ini menyajikan **ringkasan evaluasi performa model prediksi** baik secara keseluruhan maupun berbobot.
    Hasil ini menjadi dasar untuk menyimpulkan **kelayakan dan efektivitas penggunaan model** dalam konteks pengambilan keputusan berbasis data.

    Evaluasi dilakukan melalui dua pendekatan:
    - 📊 Evaluasi Global: Rata-rata dari seluruh error historis
    - ⚖️ Evaluasi Berbasis Bobot: Disesuaikan dengan kontribusi jumlah aktual masing-masing tahun

    Pendekatan berbobot selaras dengan prinsip **Knowledge-Based System (KBS)** karena mempertimbangkan dampak proporsional tiap observasi dalam sistem penalaran.
    """)

    if df_eval_total is None or df_eval_total.empty:
        st.warning("⚠️ Data evaluasi belum tersedia. Jalankan Modul Evaluasi.")
        return

    df_historis = df_eval_total[df_eval_total['Aktual'].notna()].copy()
    nama_layanan = layanan_aktif(df_eval_total)

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

    st.caption("📌 Rata-rata error historis dari model prediksi untuk setiap layanan.")

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

    st.caption("⚖️ Evaluasi berbobot yang mencerminkan dampak aktual per tahun terhadap skor global.")

    st.subheader("📝 Interpretasi dan Kesimpulan Akademik")

    kesimpulan = f"""
Model prediksi untuk layanan **{nama_layanan}** dievaluasi berdasarkan dua perspektif:

- **Global Unweighted Evaluation**:
  - MAE: **{global_mae:,.2f}**, RMSE: **{global_rmse:,.2f}**, MAPE: **{global_mape:.2f}%**
  - Kategori: **{global_kategori}** — mencerminkan rata-rata performa umum model

- **Weighted Evaluation** (berbasis bobot aktual):
  - MAE: **{mae_w:,.2f}**, RMSE: **{rmse_w:,.2f}**, MAPE: **{mape_w:.2f}%**
  - Kategori: **{kategori_w}** — memberikan gambaran yang lebih adil dan proporsional terhadap data berdampak besar

### 🔍 Analisis Akademik:
Evaluasi berbobot lebih disarankan dalam sistem berbasis pengetahuan (KBS), karena mempertimbangkan **relevansi kuantitatif setiap observasi**. Layanan dengan jumlah aktual besar secara otomatis memiliki kontribusi lebih besar terhadap performa sistem, yang mendekati cara kerja reasoning berbasis bukti (*evidence-based weighting*).

Hasil menunjukkan bahwa model Prophet memiliki performa yang **{kategori_w.lower()}**, dan layak digunakan untuk mendukung pengambilan keputusan prediktif di lingkungan layanan publik seperti DJID.

Jika dibutuhkan peningkatan akurasi, maka:
- Dapat digunakan model alternatif seperti XGBoost/LSTM untuk tren nonlinier.
- Disarankan untuk meningkatkan kualitas dan resolusi data tahunan.

Kesimpulannya, pendekatan prediktif ini mampu berfungsi sebagai komponen **inteligensi kuantitatif dalam arsitektur KBS modern**.
"""

    st.markdown(kesimpulan)
