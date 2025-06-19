# Modul_Kesimpulan.py

import streamlit as st
import pandas as pd

def modul_kesimpulan(df_prediksi):
    st.title("🧾 Modul Kesimpulan Analisis")

    st.markdown(
        """
        Modul ini menyajikan **kesimpulan otomatis** berdasarkan hasil evaluasi akurasi model prediksi
        yang dihitung dari data per layanan maupun data total tahunan.
        """
    )

    if df_prediksi is None or df_prediksi.empty:
        st.warning("⚠️ Data evaluasi tidak tersedia. Jalankan Modul Prediksi atau Evaluasi terlebih dahulu.")
        return

    # Deteksi kolom validasi & label
    validasi_col = 'Validasi Akurasi' if 'Validasi Akurasi' in df_prediksi.columns else 'Kategori Akurasi'
    label_col = 'Layanan' if 'Layanan' in df_prediksi.columns else 'Tahun'

    st.subheader("📌 Rangkuman Evaluasi")
    total = len(df_prediksi)
    sangat_akurat = len(df_prediksi[df_prediksi[validasi_col].str.contains("Sangat Akurat")])
    akurat = len(df_prediksi[df_prediksi[validasi_col].str.contains("^Akurat", regex=True)])
    cukup_akurat = len(df_prediksi[df_prediksi[validasi_col].str.contains("Cukup Akurat")])
    tidak_akurat = len(df_prediksi[df_prediksi[validasi_col].str.contains("Tidak Akurat")])

    st.write(f"- Total data evaluasi: **{total}**")
    st.write(f"- Sangat Akurat (MAPE ≤ 10%): **{sangat_akurat}**")
    st.write(f"- Akurat (MAPE ≤ 20%): **{akurat}**")
    st.write(f"- Cukup Akurat (MAPE ≤ 50%): **{cukup_akurat}**")
    st.write(f"- Tidak Akurat (MAPE > 50%): **{tidak_akurat}**")

    # Identifikasi performa terbaik dan terburuk
    best_row = df_prediksi.sort_values('MAPE (%)').iloc[0]
    worst_row = df_prediksi.sort_values('MAPE (%)').iloc[-1]

    st.subheader("🏆 Akurasi Terbaik")
    st.markdown(f"""
    - **{label_col}:** {best_row[label_col]}
    - **MAPE:** {best_row['MAPE (%)']:.2f}%
    - **Kategori:** {best_row[validasi_col]}
    """)

    st.subheader("⚠️ Akurasi Terburuk")
    st.markdown(f"""
    - **{label_col}:** {worst_row[label_col]}
    - **MAPE:** {worst_row['MAPE (%)']:.2f}%
    - **Kategori:** {worst_row[validasi_col]}
    """)

    # Kesimpulan Otomatis
    st.subheader("📝 Kesimpulan Naratif Otomatis")
    kesimpulan = f"""
Berdasarkan hasil evaluasi terhadap **{total}** entitas (berdasarkan {label_col.lower()}), 
sebanyak **{sangat_akurat + akurat} entitas ({round((sangat_akurat + akurat)/total*100, 2)}%)**
tergolong dalam akurasi baik (MAPE ≤ 20%).

Entitas dengan performa terbaik adalah **{best_row[label_col]}** dengan MAPE **{best_row['MAPE (%)']:.2f}%**,
masuk dalam kategori **{best_row[validasi_col]}**.

Sebaliknya, entitas dengan performa terburuk adalah **{worst_row[label_col]}** 
dengan MAPE **{worst_row['MAPE (%)']:.2f}%**, tergolong **{worst_row[validasi_col]}**.

Temuan ini menjadi dasar penting untuk perbaikan data historis, penyesuaian model, 
dan penentuan kebijakan berbasis data yang lebih presisi.
"""
    st.markdown(kesimpulan)
