import streamlit as st

st.set_page_config(page_title="Sistem Analisis Risiko Diabetes", layout="wide")

# ======================
# CSS Styling
# ======================
st.markdown("""
<style>
/* TITLE CARD */
.card-title {
    background-color: #0d47a1;
    color: white;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    margin-bottom: 30px;
    width: 100%;
}

/* CARD */
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    width: 100%;
}
.card h3 { margin-top: 0; font-size: 1.6rem; }
.card p, .card ul li, .card ol li {
    font-size: 1rem; 
    line-height: 1.6; 
    color: #333;
}

/* TOMBOL */
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
</style>
""", unsafe_allow_html=True)

# ======================
# HALAMAN HOME
# ======================
st.markdown("<div class='card-title'>Sistem Analisis Risiko Diabetes</div>", unsafe_allow_html=True)

# ======================
# Selamat Datang
# ======================
st.markdown("""
<div class='card'>
<h3>Selamat Datang</h3>
<p>Aplikasi ini membantu menganalisis risiko diabetes pasien menggunakan model <strong>Random Forest Classification</strong>.
Hasil analisis mencakup estimasi risiko individual, visualisasi faktor risiko utama, dan informasi statistik dataset.</p>
<p><em>Catatan:</em> Hasil ini hanya bersifat prediksi dan edukasi, <strong>tidak menggantikan diagnosis profesional.</strong></p>
</div>
""", unsafe_allow_html=True)

# ======================
# Apa itu Diabetes
# ======================
st.markdown("""
<div class='card'>
<h3>Apa itu Diabetes?</h3>
<p>Diabetes adalah kondisi kronis di mana tubuh tidak dapat mengatur kadar gula darah secara optimal.
Jika tidak terkontrol, diabetes dapat menyebabkan komplikasi serius seperti penyakit jantung, gagal ginjal, kerusakan saraf, dan gangguan penglihatan.
Deteksi dini risiko diabetes sangat penting untuk pencegahan dan pengelolaan yang tepat.</p>
</div>
""", unsafe_allow_html=True)

# ======================
# Tujuan Sistem
# ======================
st.markdown("""
<div class='card'>
<h3>Tujuan Sistem</h3>
<ul>
<li>Memberikan estimasi risiko diabetes pada pasien berdasarkan data historis.</li>
<li>Membantu tenaga medis memprioritaskan tindakan preventif.</li>
<li>Memberikan edukasi kepada pasien terkait faktor risiko utama.</li>
<li>Meningkatkan kesadaran tentang gaya hidup sehat dan kontrol gula darah.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ======================
# Fitur Sistem
# ======================
st.markdown("""
<div class='card'>
<h3>Fitur Sistem</h3>
<ul>
<li><strong>Input Dataset:</strong> Unggah dataset pasien (CSV/Excel) untuk analisis cepat.</li>
<li><strong>Preprocessing:</strong> Bersihkan data otomatis, tangani missing value, encoding, dan outlier.</li>
<li><strong>Analisis Model:</strong> Bangun model Random Forest dan evaluasi performa.</li>
<li><strong>Visualisasi Risiko:</strong> Distribusi risiko pasien, scatter plot fitur vs risiko, dan pie chart persentase risiko.</li>
<li><strong>Prediksi Data Baru:</strong> Prediksi risiko pasien baru secara langsung.</li>
<li><strong>Download Hasil:</strong> Unduh hasil analisis dalam format CSV/Excel.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ======================
# Tombol Navigasi (Center)
# ======================
col1, col2, col3 = st.columns([1, 10, 1])

with col2:
    if st.button("Mulai Analisis / Input Dataset", key="start_analysis", use_container_width=True):
        st.switch_page("pages/1_Input_Dataset.py")
