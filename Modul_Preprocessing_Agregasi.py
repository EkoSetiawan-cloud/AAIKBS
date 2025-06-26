import streamlit as st
import pandas as pd

def modul_preprocessing_agregasi(df):
    st.title("🧹 Modul Preprocessing: Agregasi Data Historis")

    st.markdown("""
    Modul ini bertugas melakukan **Pembersihan dan Transformasi** dataset awal menjadi bentuk yang siap untuk:
    - 📊 Visualisasi tren
    - 🔗 Klasterisasi pola layanan
    - 🔮 Prediksi layanan mendatang

    ### 🔧 Proses Preprocessing Data:
    1. Pembersihan kolom kosong & duplikat
    2. Deteksi otomatis kolom tahun
    3. Transformasi format *wide* → *long*
    4. Agregasi total per tahun
    """)

    if df is None:
        st.warning("⚠️ Silakan unggah dataset terlebih dahulu di Modul Input.")
        return None

    # Tampilkan dimensi awal
    st.subheader("🔍 Dimensi Awal Dataset")
    st.info(f"Dataset terdiri dari **{df.shape[0]} baris** dan **{df.shape[1]} kolom** sebelum dibersihkan.")

    # Pembersihan awal
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.dropna(how='all', axis=1)

    st.success("✅ Setelah membersihkan duplikat dan kolom kosong:")
    st.dataframe(df_cleaned.head())

    # Deteksi kolom tahun
    year_columns = [col for col in df_cleaned.columns if str(col).isdigit()]
    if not year_columns:
        st.error("❌ Kolom tahun tidak ditemukan. Pastikan kolom tahun bernama 2019, 2020, dst.")
        return None

    # Pilih kolom identitas layanan
    layanan_col = st.selectbox("🧾 Pilih kolom identitas Layanan", options=df_cleaned.columns)

    # Transformasi wide → long
    df_long = df_cleaned.melt(
        id_vars=layanan_col,
        value_vars=year_columns,
        var_name='Tahun',
        value_name='Jumlah'
    )
    df_long['Tahun'] = df_long['Tahun'].astype(int)
    df_long = df_long.dropna()

    st.subheader("📊 Dataset Setelah Transformasi Wide → Long")
    st.dataframe(df_long.head())

    st.caption("""
    ✅ Sekarang data sudah berbentuk long format (1 baris = 1 kombinasi Layanan-Tahun). 
    Format ini **esensial untuk klastering, prediksi, dan evaluasi model**.
    """)

    # ✅ Agregasi total per tahun
    st.subheader("📈 Total Jumlah Layanan DJID per Tahun")
    df_total = df_long.groupby('Tahun')['Jumlah'].sum().reset_index()
    df_total.columns = ['Tahun', 'Total Jumlah Layanan']
    st.dataframe(df_total)

    st.caption("""
    📌 Agregasi ini berguna untuk melihat **tren makro jumlah layanan** dari tahun ke tahun secara nasional.
    """)

    return df_long
