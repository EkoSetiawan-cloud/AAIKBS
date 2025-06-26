import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st

# Import semua modul
from Modul_Input import modul_input_page
from Modul_Preprocessing_Agregasi import modul_preprocessing_agregasi
from Modul_Clustering_Tren import modul_clustering_tren
from Modul_Prediksi import modul_prediksi
from Modul_Evaluasi import modul_evaluasi
from Modul_Kesimpulan import modul_kesimpulan

# Konfigurasi layout halaman
st.set_page_config(page_title="KBS - Prediksi Layanan DJID", layout="wide")

# Judul & Navigasi Sidebar
with st.sidebar:
    st.markdown("## 📚 Prediksi Layanan DJID")
    modul = st.radio("Pilih Modul", (
        "Input Dataset",
        "Preprocessing Data",
        "Model Clustering Tren",
        "Model Prediksi Layanan",
        "Evaluasi Model",
        "Kesimpulan"
    ))

# Inisialisasi session state jika belum ada
state = st.session_state
for key in ['df_raw', 'df_agregasi', 'df_clustered', 'df_prediksi',
            'df_eval_total', 'df_clustered_tren']:
    if key not in state:
        state[key] = None

# Routing antar modul
if modul == "Input Dataset":
    state.df_raw = modul_input_page()

elif modul == "Preprocessing Data":
    if state.df_raw is not None:
        state.df_agregasi = modul_preprocessing_agregasi(state.df_raw)
    else:
        st.warning("⚠️ Silakan unggah dataset terlebih dahulu di Input Dataset.")

elif modul == "Model Clustering Tren":
    if state.df_agregasi is not None:
        state.df_clustered_tren = modul_clustering_tren(state.df_agregasi)
    else:
        st.warning("⚠️ Silakan jalankan Preprocessing Data terlebih dahulu.")

elif modul == "Model Prediksi Layanan":
    if state.df_agregasi is not None:
        df_pred, df_eval = modul_prediksi(state.df_agregasi)
        state.df_prediksi = df_pred
        state.df_eval_total = df_eval
    else:
        st.warning("⚠️ Silakan jalankan Preprocessing Data terlebih dahulu.")

elif modul == "Evaluasi Model":
    if state.df_prediksi is not None:
        df_eval_total = modul_evaluasi(state.df_prediksi)
        state.df_eval_total = df_eval_total
    else:
        st.warning("⚠️ Data prediksi tidak ditemukan. Jalankan Model Prediksi Layanan terlebih dahulu.")

elif modul == "Kesimpulan":
    if state.df_eval_total is not None:
        modul_kesimpulan(state.df_eval_total)
    else:
        st.warning("⚠️ Data evaluasi tidak ditemukan. Jalankan Evaluasi Model terlebih dahulu.")
