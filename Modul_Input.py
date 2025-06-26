import streamlit as st
import pandas as pd

def modul_input_page():
    st.title("📥 Modul Input: Upload Dataset Historis Layanan DJID")

    st.markdown("""
    Modul ini digunakan untuk mengunggah **dataset historis layanan DJID** dalam format `.xlsx`.

    ### 🎯 Tujuan:
    - Menyediakan data awal untuk seluruh proses prediksi dan evaluasi.
    - Dataset ini akan digunakan oleh modul preprocessing, clustering, hingga prediksi.

    > **Struktur ideal dataset:**
    - Kolom pertama: Nama layanan (misalnya: `Layanan DJID`)
    - Kolom selanjutnya: Tahun (misal: `2019`, `2020`, ..., `2024`)
    """)

    uploaded_file = st.file_uploader("📂 Unggah Dataset (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("✅ Dataset berhasil diunggah!")
            st.markdown("### 👀 Pratinjau 5 Baris Pertama Dataset")
            st.dataframe(df.head())

            st.caption("""
            ✅ Data akan disimpan di memory (session_state) dan digunakan oleh modul-modul berikutnya.

            📌 Pastikan:
            - Tidak ada baris duplikat
            - Tahun ditulis dalam format numerik (`2021`, bukan `Thn 2021`)
            """)

            st.session_state.df_raw = df  # Simpan ke session_state

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    else:
        st.info("ℹ️ Silakan unggah file Excel terlebih dahulu.")

    return st.session_state.get('df_raw', None)
