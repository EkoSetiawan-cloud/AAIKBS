# Modul_Input.py

import streamlit as st
import pandas as pd

def modul_input_page():
    st.title("📥 Modul Input: Upload Dataset Historis Layanan DJID")

    st.markdown(
        """
        Modul ini digunakan untuk mengunggah dataset yang akan digunakan dalam proses Knowledge-Based System.
        Pastikan dataset yang diunggah bertipe `.xlsx` dan memiliki struktur kolom yang relevan dengan Layanan DJID.
        """
    )

    uploaded_file = st.file_uploader("Unggah Dataset (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("✅ Dataset berhasil diunggah!")
            st.write("### Tampilan 5 Row Dataset :")
            st.dataframe(df.head())

            # ✅ Simpan ke session_state
            st.session_state.df_raw = df

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.info("Silakan unggah file Excel terlebih dahulu.")

    # ✅ Kembalikan dari session_state
    return st.session_state.get('df_raw', None)
