import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualisasi Risiko Diabetes", layout="wide")

# ======================================================
# CSS Styling
# ======================================================
st.markdown("""
<style>
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
    color: #ffffff !important;
    box-shadow: 0 6px 16px rgba(13,71,161,0.18);
}
.card { background-color: #ffffff; border-radius: 12px; padding: 20px;
       box-shadow: 0 6px 18px rgba(0,0,0,0.08); margin-bottom: 20px; }
.card h3 { margin-top: 0; font-size: 1.5rem; }
.card p { font-size: 1rem; line-height: 1.6; color: #333; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# Header Utama
# ======================================================
st.markdown("<div class='main-title'>Sistem Analisis Risiko Diabetes</div>", unsafe_allow_html=True)

# ======================================================
# Card Deskripsi
# ======================================================
st.markdown("""
<div class='card'>
<h3>Visualisasi Risiko Pasien</h3>
<p>Visualisasi ini menampilkan pasien berdasarkan tingkat risiko diabetes.</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
if "df_risk" not in st.session_state:
    st.error("Data risiko tidak ditemukan. Jalankan halaman Analisis Data terlebih dahulu.")
    st.stop()

df = st.session_state["df_risk"].copy()
X_columns = st.session_state["X_columns"]

# Kolom visual (tanpa ID)
visual_cols = [c for c in X_columns if c.lower() != "id"]

# ======================================================
# NUMERIK & KATEGORIKAL
# ======================================================

# ----- NUMERIK -----
numerical_cols = df[visual_cols].select_dtypes(include=np.number).columns.tolist()

# ----- KATEGORIKAL -----
categorical_cols = []
if "df_original" in st.session_state:
    df_raw = st.session_state["df_original"].copy()

    # hanya tempel risk_level (untuk visualisasi)
    df_raw["risk_level"] = df["risk_level"]

    # Ambil hanya kolom object dari data original
    categorical_cols = df_raw.select_dtypes(include="object").columns.tolist()

    # Pastikan risk_level tidak terdeteksi sebagai kategorikal
    categorical_cols = [c for c in categorical_cols if c != "risk_level"]

else:
    st.warning("Data asli (df_original) tidak ditemukan.")
    categorical_cols = []
# ======================================================
# 1. Pie Chart Risiko
# ======================================================
st.markdown("""
<div class='card'>
    <h3>Distribusi Risiko Pasien</h3>
    <p>Visualisasi ini menampilkan persentase pasien dalam kategori risiko diabetes (Rendah, Sedang, Tinggi). 
    Berguna untuk memahami sebaran risiko secara keseluruhan di populasi pasien.</p>
</div>
""", unsafe_allow_html=True)

risk_counts = df["risk_level"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0).reset_index()
risk_counts.columns = ["Risk Level", "Jumlah"]

fig_pie = px.pie(
    risk_counts, values="Jumlah", names="Risk Level",
    hole=0.3,
    color="Risk Level",
    color_discrete_map={"Rendah": "green", "Sedang": "yellow", "Tinggi": "red"}
)
st.plotly_chart(fig_pie, use_container_width=True)


# ======================================================
# 2. Scatter Plot Numerik vs Risiko
# ======================================================
if numerical_cols:
    st.markdown("""
    <div class='card'>
        <h3>Scatter Plot Fitur Numerik vs Skor Risiko</h3>
        <p>Visualisasi ini menampilkan hubungan antara satu fitur numerik dengan skor risiko pasien. 
        Berguna untuk melihat pola atau korelasi antara fitur numerik tertentu dan tingkat risiko diabetes.</p>
    </div>
    """, unsafe_allow_html=True)

    selected_feature = st.selectbox("Pilih fitur numerik:", numerical_cols)

    hover_candidates = [c for c in numerical_cols if c != selected_feature]
    hover_data = hover_candidates[:6]

    fig_scatter = px.scatter(
        df,
        x=selected_feature,
        y="risk_score",
        color="risk_level",
        hover_data=hover_data,
        color_discrete_map={"Rendah": "green", "Sedang": "yellow", "Tinggi": "red"},
        labels={"risk_score": "Skor Risiko"},
        title=f"{selected_feature} vs Skor Risiko"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ======================================================
# 3. Distribusi Probabilitas Risiko (Categorical)
# ======================================================
if categorical_cols:
    st.markdown("""
    <div class='card'>
        <h3>Distribusi Probabilitas Risiko berdasarkan Fitur Kategorikal</h3>
        <p>Visualisasi ini menunjukkan probabilitas risiko diabetes di setiap kategori dari fitur kategorikal. 
        Berguna untuk mengidentifikasi kategori yang memiliki risiko tinggi atau rendah secara cepat.</p>
    </div>
    """, unsafe_allow_html=True)

    selected_cat = st.selectbox("Pilih fitur kategorikal:", categorical_cols)

    prob_df = (
        df_raw.groupby([selected_cat, "risk_level"])
        .size()
        .reset_index(name="count")
    )

    prob_df["probability"] = prob_df.groupby(selected_cat)["count"].transform(lambda x: x / x.sum())

    fig_cat = px.bar(
        prob_df,
        x=selected_cat,
        y="probability",
        color="risk_level",
        text=prob_df["probability"].apply(lambda x: f"{x:.2f}"),
        barmode="stack",
        color_discrete_map={"Rendah": "green", "Sedang": "yellow", "Tinggi": "red"},
        labels={"probability": "Probabilitas Risiko"},
        title=f"Distribusi Probabilitas Risiko berdasarkan '{selected_cat}'"
    )
    fig_cat.update_layout(yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig_cat, use_container_width=True)

# ======================================================
# Tombol Switch Halaman
# ======================================================
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    if st.button("Prediksi Data Baru", use_container_width=True):
        st.switch_page("Pages/5_Prediksi_Data_Baru.py")
