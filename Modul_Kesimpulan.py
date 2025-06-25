# Modul_Kesimpulan.py

import streamlit as st
import pandas as pd

def modul_kesimpulan(df_eval_total):
    st.title("🧾 Modul Kesimpulan Evaluasi Total per Tahun")

    if df_eval_total is None or df_eval_total.empty:
        st.warning("⚠️ Data evaluasi belum tersedia. Jalankan Modul Evaluasi Total terlebih dahulu.")
        return

    st.subheader("📌 Rangkuman Evaluasi")
    total = len(df_eval_total)
    sangat_akurat = len(df_eval_total[df_eval_total['Validasi Akurasi'].str.contains("Sangat Akurat")])
    akurat = len(df_eval_total[df_eval_total['Validasi Akurasi'].str.contains("^Akurat", regex=True)])
    cukup_akurat = len(df_eval_total[df_eval_total['Validasi Akurasi'].str.contains("Cukup Akurat")])
    tidak_akurat = len(df_eval_total[df_eval_total['Validasi Akurasi'].str.contains("Tidak Akurat")])

    st.write(f"- Total tahun evaluasi: **{total}**")
    st.write(f"- Sangat Akurat (MAPE ≤ 10%): **{sangat_akurat} tahun**")
    st.write(f"- Akurat (10% < MAPE ≤ 20%): **{akurat} tahun**")
    st.write(f"- Cukup Akurat (20% < MAPE ≤ 50%): **{cukup_akurat} tahun**")
    st.write(f"- Tidak Akurat (MAPE > 50%): **{tidak_akurat} tahun**")

    # Tahun terbaik dan terburuk
    best_row = df_eval_total.sort_values('MAPE (%)').iloc[0]
    worst_row = df_eval_total.sort_values('MAPE (%)').iloc[-1]

    st.subheader("🏆 Tahun dengan Akurasi Terbaik")
    st.markdown(f"""
    - **Tahun:** {best_row['Tahun']}
    - **MAPE:** {best_row['MAPE (%)']:.2f}%
    - **Kategori:** {best_row['Validasi Akurasi']}
    """)

    st.subheader("⚠️ Tahun dengan Akurasi Terburuk")
    st.markdown(f"""
    - **Tahun:** {worst_row['Tahun']}
    - **MAPE:** {worst_row['MAPE (%)']:.2f}%
    - **Kategori:** {worst_row['Validasi Akurasi']}
    """)

    st.subheader("📝 Kesimpulan Naratif Otomatis")
    persentase_baik = round(((sangat_akurat + akurat) / total) * 100, 2)
    kesimpulan = f"""
Dari total **{total} tahun evaluasi**, sebanyak **{sangat_akurat + akurat} tahun ({persentase_baik}%)** 
berada dalam kategori **akurat atau sangat akurat** berdasarkan nilai MAPE.

Tahun dengan performa terbaik adalah **{best_row['Tahun']}** dengan nilai MAPE hanya **{best_row['MAPE (%)']:.2f}%**
dan diklasifikasikan sebagai **{best_row['Validasi Akurasi']}**.

Sebaliknya, tahun dengan performa prediksi terburuk adalah **{worst_row['Tahun']}**, 
dengan MAPE mencapai **{worst_row['MAPE (%)']:.2f}%** dan dikategorikan sebagai **{worst_row['Validasi Akurasi']}**.

Hasil evaluasi ini menunjukkan bahwa model Prophet dapat memberikan hasil prediksi yang sangat baik,
dan layak dijadikan dasar kebijakan dan perencanaan layanan DJID ke depan.
"""
    st.markdown(kesimpulan)
