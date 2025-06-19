# kbs.py

import streamlit as st

# Import semua modul yang sudah dibuat
from Modul_Input import modul_input_page
from Modul_Preprocessing_Agregasi import modul_preprocessing_agregasi
from Modul_Clustering import modul_clustering
from Modul_Prediksi import modul_prediksi
from Modul_Prediksi_Total import modul_prediksi_total
from Modul_Kesimpulan import modul_kesimpulan

# Konfigurasi halaman
st.set_page_config(page_title="KBS - Prediksi Layanan DJID", layout="wide")
st.sidebar.title("📚 Navigasi Modul")

# Navigasi
modul = st.sidebar.radio("Pilih Modul", (
    "Modul Input",
    "Modul Preprocessing",
    "Modul Clustering",
    "Modul Prediksi",
    "Modul Prediksi Total",
    "Modul Kesimpulan"
))

# Session State untuk menyimpan data antar modul
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_agregasi' not in st.session_state:
    st.session_state.df_agregasi = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'df_prediksi' not in st.session_state:
    st.session_state.df_prediksi = None
if 'df_prediksi_total' not in st.session_state:
    st.session_state.df_prediksi_total = None
if 'df_gabungan' not in st.session_state:
    st.session_state.df_gabungan = None

# Routing antar modul
if modul == "Modul Input":
    st.session_state.df_raw = modul_input_page()

elif modul == "Modul Preprocessing":
    if st.session_state.df_raw is not None:
        st.session_state.df_agregasi = modul_preprocessing_agregasi(st.session_state.df_raw)
    else:
        st.warning("⚠️ Silakan unggah dataset terlebih dahulu di Modul Input.")

elif modul == "Modul Clustering":
    if st.session_state.df_agregasi is not None:
        st.session_state.df_clustered = modul_clustering(st.session_state.df_agregasi)
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Prediksi":
    if st.session_state.df_agregasi is not None:
        df_pred, df_eval = modul_prediksi(st.session_state.df_agregasi)
        st.session_state.df_prediksi = df_pred
        st.session_state.df_prediksi_total = df_eval
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Prediksi Total":
    if st.session_state.df_agregasi is not None:
        modul_prediksi_total(st.session_state.df_agregasi)
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Kesimpulan":
    if st.session_state.df_prediksi_total is not None:
        modul_kesimpulan(st.session_state.df_prediksi_total)
    else:
        st.warning("⚠️ Data evaluasi tidak ditemukan. Jalankan Modul Evaluasi terlebih dahulu.")