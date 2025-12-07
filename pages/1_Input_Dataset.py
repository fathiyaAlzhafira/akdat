import streamlit as st
import pandas as pd

# -------------------------------------------------------
# Konfigurasi halaman utama
# -------------------------------------------------------
st.set_page_config(
    page_title="Input Dataset Diabetes",
    layout="wide"
)

# -------------------------------------------------------
# CSS Styling untuk tampilan yang lebih rapi
# -------------------------------------------------------
st.markdown(
    """
    <style>
    /* CARD JUDUL UTAMA */
    .main-title {
        background-color: #0d47a1;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }

    /* CARD KONTEN */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .card h3 {
        margin-top: 0;
        font-size: 1.5rem;
    }
    .card p, .card ul li, .card ol li {
        font-size: 1rem;
        line-height: 1.6;
        color: #333;
    }

    /* FILE UPLOADER */
    .stFileUploader > label {
        font-size: 1rem !important;
        font-weight: 600;
        color: #0d47a1 !important;
    }

    /* BUTTON */
    div.stButton > button {
        font-weight: 700;
        color: #0d47a1 !important;
        background: white !important;
        border: 2px solid #0d47a1 !important;
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(13,71,161,0.12);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    div.stButton > button:hover {
        background-color: #0d47a1 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Judul halaman
# -------------------------------------------------------
st.markdown(
    "<div class='main-title'>Sistem Analisis Risiko Diabetes</div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Card deskripsi upload dataset
# -------------------------------------------------------
st.markdown(
    """
    <div class='card'>
      <h3>1. Upload Dataset</h3>
      <p>
        Unggah dataset pasien dalam format <strong>CSV, XLSX, atau XLS</strong>
        untuk dianalisis menggunakan model <strong>Random Forest Classification</strong>.
      </p>
      <p>
        Pastikan dataset memiliki satu kolom target yang menunjukkan status diabetes
        (misalnya: <em>Diabetes / Non-Diabetes</em>, <em>Outcome</em>, atau label lain).
      </p>
      <p>
        Dataset ini akan digunakan pada tahap preprocessing dan analisis model
        untuk menghitung risiko diabetes setiap pasien.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Komponen upload file
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    label="Pilih file dataset",
    type=["csv", "xlsx", "xls"],
    key="file_uploader",
    help="Unggah file CSV, XLSX, atau XLS berisi data pasien.",
)

# -------------------------------------------------------
# Jika file berhasil di-upload
# -------------------------------------------------------
if uploaded_file is not None:
    # Baca file sesuai ekstensi
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Simpan dataset ke session_state agar bisa dipakai di halaman lain
    st.session_state["dataset"] = df

    # Card informasi singkat dataset
    st.markdown(
        f"""
        <div class='card'>
          <h3>Informasi Dataset</h3>
          <ul>
            <li>Total baris: {len(df)}</li>
            <li>Total kolom: {len(df.columns)}</li>
            <li>Kolom tersedia: {', '.join(df.columns)}</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preview beberapa baris pertama
    st.subheader("Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------------------------------------------
    # Deteksi otomatis calon kolom target (jika ada)
    # ---------------------------------------------------
    detected_target = None
    candidate_names = [
        "diabetes",
        "diagnosed_diabetes",
        "outcome",
        "target",
        "label",
        "class",
    ]

    for col in df.columns:
        if col.lower() in candidate_names:
            detected_target = col
            break

    # ---------------------------------------------------
    # Pilih kolom target secara manual (dengan default
    # hasil deteksi otomatis jika ditemukan)
    # ---------------------------------------------------
    target_choice = st.selectbox(
        "Pilih kolom target (label yang akan diprediksi):",
        options=list(df.columns),
        index=df.columns.get_loc(detected_target) if detected_target else 0,
    )

    # Simpan nama kolom target ke session_state
    st.session_state["target_col"] = target_choice
    st.info(f"Kolom target disimpan: **{target_choice}**")

    # ---------------------------------------------------
    # Tombol untuk lanjut ke halaman preprocessing
    # ---------------------------------------------------
    if st.button("Lanjut ke Preprocessing", use_container_width=True):
        st.switch_page("pages/2_Preprocessing.py")

# -------------------------------------------------------
# Jika belum ada file yang di-upload
# -------------------------------------------------------
else:
    st.info("Silakan upload file CSV atau Excel terlebih dahulu untuk melanjutkan.")
