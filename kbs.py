import streamlit as st

# Import semua modul
from Modul_Input import modul_input_page
from Modul_Preprocessing_Agregasi import modul_preprocessing_agregasi
from Modul_Clustering import modul_clustering
from Modul_Prediksi import modul_prediksi
from Modul_Prediksi_Total import modul_prediksi_total
from Modul_Evaluasi_Total import modul_evaluasi_total
from Modul_Kesimpulan import modul_kesimpulan

# Konfigurasi layout halaman
st.set_page_config(page_title="KBS - Prediksi Layanan DJID", layout="centered")

# Judul & Navigasi Tengah
st.markdown("<h2 style='text-align:center;'>📚 Navigasi Modul</h2>", unsafe_allow_html=True)
modul = st.radio("Pilih Modul", (
    "Modul Input",
    "Modul Preprocessing",
    "Modul Clustering",
    "Modul Prediksi",
    "Modul Prediksi Total",
    "Modul Evaluasi Total",
    "Modul Kesimpulan"
))


# Inisialisasi session state jika belum ada
state = st.session_state
for key in ['df_raw', 'df_agregasi', 'df_clustered', 'df_prediksi',
            'df_prediksi_total', 'df_eval_total', 'df_gabungan']:
    if key not in state:
        state[key] = None

# Routing antar modul
if modul == "Modul Input":
    state.df_raw = modul_input_page()

elif modul == "Modul Preprocessing":
    if state.df_raw is not None:
        state.df_agregasi = modul_preprocessing_agregasi(state.df_raw)
    else:
        st.warning("⚠️ Silakan unggah dataset terlebih dahulu di Modul Input.")

elif modul == "Modul Clustering":
    if state.df_agregasi is not None:
        state.df_clustered = modul_clustering(state.df_agregasi)
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Prediksi":
    if state.df_agregasi is not None:
        df_pred, df_eval = modul_prediksi(state.df_agregasi)
        state.df_prediksi = df_pred
        # Catatan: df_eval ini berbasis per layanan
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Prediksi Total":
    if state.df_agregasi is not None:
        df_total = modul_prediksi_total(state.df_agregasi)
        state.df_prediksi_total = df_total
    else:
        st.warning("⚠️ Silakan jalankan Modul Preprocessing terlebih dahulu.")

elif modul == "Modul Evaluasi Total":
    if state.df_prediksi_total is not None:
        df_eval_total = modul_evaluasi_total(state.df_prediksi_total)
        state.df_eval_total = df_eval_total  # <--- return disimpan untuk kesimpulan
    else:
        st.warning("⚠️ Data prediksi total tidak ditemukan. Jalankan Modul Prediksi Total terlebih dahulu.")

elif modul == "Modul Kesimpulan":
    if state.df_eval_total is not None:
        modul_kesimpulan(state.df_eval_total)
    else:
        st.warning("⚠️ Data evaluasi total tidak ditemukan. Jalankan Modul Evaluasi Total terlebih dahulu.")
