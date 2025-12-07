import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------
# Cek ketersediaan library imbalanced-learn (SMOTE, dsb)
# -------------------------------------------------------
SMOTE_AVAILABLE = True
SMOTEENN_AVAILABLE = True
SMOTETOMEK_AVAILABLE = True
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
except Exception:
    SMOTE_AVAILABLE = False
    SMOTEENN_AVAILABLE = False
    SMOTETOMEK_AVAILABLE = False


# -------------------------------------------------------
# Helper untuk membuat figure kecil (hemat ruang UI)
# -------------------------------------------------------
PLOT_W_COMPACT = 4.0
PLOT_H_COMPACT = 2.5


def make_fig(w: float = PLOT_W_COMPACT, h: float = PLOT_H_COMPACT):
    """Buat figure matplotlib kecil dengan ukuran default."""
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# -------------------------------------------------------
# Konfigurasi halaman + CSS
# -------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Analisis Risiko Diabetes — Random Forest",
)

st.markdown(
    """
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
    .section-title { 
        font-size:1.4rem; 
        font-weight:700; 
        margin:24px 0 12px 0; 
        color: #1e293b;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .eval-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
    }
    .eval-card h3 {
        margin: 0 0 8px 0;
        font-size: 1.5rem;
    }
    .eval-card p {
        margin: 0;
        opacity: 0.95;
    }
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight:700;
    }
    .low { background: #16a34a; }
    .mid { background: #f59e0b; color: #111827; }
    .high { background: #dc2626; }
    .small-note { 
        font-size:0.9rem; 
        color:#475569; 
        margin-top: 8px;
    }
    /* Full-width buttons */
    div.stButton, div.stDownloadButton {
        width: 100% !important;
    }
    div.stButton > button, div.stDownloadButton > button {
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
    .stSelectbox, .stSlider, .stNumberInput, .stRadio { 
        width: 100% !important; 
    }
    .table-container .dataframe { 
        max-height: 520px; 
        overflow:auto; 
    }
    /* Plot lebih kecil dan berada di tengah */
    .stpyplot {
        max-width: 500px;
        margin: 1.5rem auto;
        display: flex;
        justify-content: center;
    }
    .plot-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul utama halaman
st.markdown(
    "<div class='main-title'>Sistem Analisis Risiko Diabetes</div>",
    unsafe_allow_html=True,
)

# Card pengantar analisis
st.markdown(
    """
    <div class='card'>
        <h3>3. Analisis Data dengan Algoritma Random Forest</h3>
        <p>
            Di halaman ini model <strong>Random Forest Classification</strong>
            dilatih untuk memprediksi risiko diabetes pasien.
        </p>
        <ul>
            <li>
                <strong>Teknik Balancing:</strong>
                Pilih metode penyeimbangan dataset jika kelas target tidak seimbang
                (misal: SMOTE, SMOTEENN, SMOTETomek, atau Class Weight).
            </li>
            <li>
                <strong>Pilihan Parameter:</strong>
                <ul>
                    <li><em>Basic</em>: menggunakan parameter default.</li>
                    <li><em>Advanced</em>: mengatur sendiri n_estimators, max_depth, dsb.</li>
                </ul>
            </li>
            <li>
                <strong>Latih Model:</strong>
                Setelah konfigurasi selesai, klik
                <strong>Latih & Evaluasi Model</strong>.
            </li>
        </ul>
        <p>
            Output mencakup akurasi, precision, recall, F1-score, kurva ROC,
            PR-curve, feature importance, serta tabel risiko tiap pasien.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Cek data dari tahap preprocessing
# -------------------------------------------------------
if "processed_data" not in st.session_state:
    st.error(
        "Data hasil preprocessing tidak ditemukan di "
        "`st.session_state['processed_data']`. "
        "Jalankan halaman Preprocessing terlebih dahulu."
    )
    st.stop()

df = st.session_state["processed_data"].copy()

if "target_col" not in st.session_state:
    st.error("Kolom target belum diset di session. Pilih target di halaman Input Dataset.")
    st.stop()

target_col = st.session_state["target_col"]
if target_col not in df.columns:
    st.error(f"Kolom target '{target_col}' tidak ditemukan di data.")
    st.stop()

# -------------------------------------------------------
# Ringkasan cepat dataset
# -------------------------------------------------------
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown(
        f"<div class='card'><strong>Records</strong>"
        f"<div style='font-size:20px'>{df.shape[0]:,}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='card'><strong>Features</strong>"
        f"<div style='font-size:20px'>{df.shape[1]}</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='card'><strong>Target</strong>"
        f"<div style='font-size:16px'>{target_col}</div></div>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Pilihan teknik balancing & parameter model
# -------------------------------------------------------
st.markdown(
    '<div class="section-title">1) Penyeimbangan & Parameter Model</div>',
    unsafe_allow_html=True,
)

balancing_method = st.selectbox(
    "Pilih teknik balancing:",
    options=["None", "SMOTE", "SMOTEENN", "SMOTETomek", "Class Weight (auto)"],
    index=2 if SMOTEENN_AVAILABLE else 1,
)

# Info ketersediaan metode balancing
if balancing_method == "SMOTE" and not SMOTE_AVAILABLE:
    st.warning("SMOTE tidak tersedia di environment ini. Pilih opsi lain.")
if balancing_method == "SMOTEENN" and not SMOTEENN_AVAILABLE:
    st.warning("SMOTEENN tidak tersedia. Pilih SMOTETomek / SMOTE / Class Weight.")
if balancing_method == "SMOTETomek" and not SMOTETOMEK_AVAILABLE:
    st.warning("SMOTETomek tidak tersedia. Pilih SMOTEENN / SMOTE / Class Weight.")

# Penjelasan singkat tiap metode balancing
if balancing_method == "None":
    st.info(
        "**None**: Data digunakan apa adanya. Cocok jika dataset sudah seimbang "
        "atau ingin membandingkan performa tanpa balancing."
    )
elif balancing_method == "SMOTE":
    st.info(
        "**SMOTE (Synthetic Minority Oversampling Technique)**: "
        "Membuat sampel sintetis untuk kelas minoritas dengan interpolasi. "
        "Tidak menghapus data mayoritas."
    )
elif balancing_method == "SMOTEENN":
    st.info(
        "**SMOTEENN**: SMOTE (oversampling) + ENN (undersampling). "
        "Menambah sampel minoritas dan menghapus sampel mayoritas yang bising."
    )
elif balancing_method == "SMOTETomek":
    st.info(
        "**SMOTETomek**: SMOTE (oversampling) + Tomek Links (undersampling). "
        "Membersihkan batas antara kelas mayoritas dan minoritas."
    )
elif balancing_method == "Class Weight (auto)":
    st.info(
        "**Class Weight (auto)**: Memberi bobot lebih besar pada kelas minoritas "
        "saat training tanpa mengubah jumlah baris data."
    )

# Mode pengaturan parameter
mode = st.radio("Mode parameter:", ["Basic", "Advanced"], horizontal=True)

if mode == "Basic":
    # Mode sederhana: hanya ubah test_size
    test_size = st.slider(
        "Test size (%) - persentase data yang dipakai untuk TEST",
        5,
        50,
        20,
    )
    params = {
        "n_estimators": 120,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "test_size": test_size,
    }
else:
    # Mode lanjutan: semua hyperparameter bisa diatur
    st.markdown(
        "Pilih parameter model (advanced). "
        "Test size diinterpretasikan sebagai persen dari data yang digunakan "
        "sebagai test set."
    )

    na, nb = st.columns(2)
    with na:
        n_estimators = st.number_input("n_estimators", 50, 1000, 200)
        max_depth = st.number_input("max_depth (0 = None)", 0, 100, 0)
        min_samples_split = st.number_input("min_samples_split", 2, 50, 2)
    with nb:
        min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1)
        bootstrap = st.selectbox("bootstrap", [True, False])
        test_size = st.slider(
            "Test size (%) - persentase data untuk TEST",
            5,
            50,
            20,
        )

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": None if int(max_depth) == 0 else int(max_depth),
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "bootstrap": bool(bootstrap),
        "test_size": int(test_size),
    }

    # Penjelasan parameter model
    st.markdown("### Penjelasan Parameter Model")
    st.markdown(
        """
        - **n_estimators**: Jumlah pohon dalam Random Forest. 
          Lebih banyak pohon → potensi akurasi naik, tetapi waktu training lebih lama.  
        - **max_depth**: Kedalaman maksimal pohon (0 = tidak dibatasi). 
          Membatasi kedalaman membantu mencegah overfitting.  
        - **min_samples_split**: Minimum sampel di sebuah node sebelum dibagi.  
        - **min_samples_leaf**: Minimum sampel di daun (leaf). Nilai lebih besar → model lebih sederhana.  
        - **bootstrap**: Jika `True`, tiap pohon dilatih dengan sampel bootstrap (dengan pengambilan ulang).  
        - **test_size**: Persentase data yang digunakan sebagai test set (20–30% biasanya cukup).  
        """
    )

# -------------------------------------------------------
# Tombol Latih & Evaluasi Model
# -------------------------------------------------------
# Dibuat di tengah dengan lebar penuh
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    do_train = st.button("Latih & Evaluasi Model", use_container_width=True)

# -------------------------------------------------------
# Proses training + evaluasi
# -------------------------------------------------------
if do_train:
    # Bersihkan model lama jika ada
    st.session_state.pop("trained_model", None)
    st.session_state.pop("df_risk", None)

    start_time = time.time()

    # Pisah fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"] / 100,
        stratify=y,
        random_state=42,
    )

    # Terapkan teknik balancing jika dipilih
    cw = None
    if balancing_method == "SMOTE" and SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        st.success("SMOTE diterapkan pada training set.")
    elif balancing_method == "SMOTEENN" and SMOTEENN_AVAILABLE:
        smenn = SMOTEENN(random_state=42)
        X_train, y_train = smenn.fit_resample(X_train, y_train)
        st.success("SMOTEENN diterapkan pada training set.")
    elif balancing_method == "SMOTETomek" and SMOTETOMEK_AVAILABLE:
        smt = SMOTETomek(random_state=42)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        st.success("SMOTETomek diterapkan pada training set.")
    elif balancing_method == "Class Weight (auto)":
        cw = "balanced"
        st.info("class_weight='balanced' akan digunakan saat training.")
    else:
        st.info("Tidak ada teknik balancing tambahan (None).")

    # Inisialisasi model Random Forest
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        bootstrap=params["bootstrap"],
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )

    # Training
    with st.spinner("Melatih model..."):
        model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    st.success(f"Model selesai dilatih dalam {elapsed:.1f} detik.")

    # Simpan ke session_state untuk digunakan di halaman lain
    st.session_state["trained_model"] = model
    st.session_state["model"] = model             # alias untuk halaman Visualisasi
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["processed_data"] = df       # data lengkap setelah preprocessing
    st.session_state["X_columns"] = list(X.columns)
    st.session_state["final_feature_names"] = list(X.columns)

    # Hitung skor risiko untuk seluruh data (pakai predict_proba)
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(df.drop(columns=[target_col]))[:, 1]
        df_risk = df.copy()
        df_risk["risk_score"] = probs_all
        df_risk["risk_level"] = df_risk["risk_score"].apply(
            lambda p: "Rendah" if p < 0.33 else "Sedang" if p < 0.66 else "Tinggi"
        )
        st.session_state["df_risk"] = df_risk

        # Simpan metrik global sederhana (untuk halaman Visualisasi)
        try:
            y_pred_all = model.predict(X)
            acc_all = accuracy_score(df[target_col], y_pred_all)
        except Exception:
            acc_all = None

        st.session_state["metrics"] = {
            "accuracy": float(acc_all) * 100 if acc_all is not None else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    else:
        st.warning("Model tidak mendukung predict_proba → risk score tidak dihitung.")

# -------------------------------------------------------
# Evaluasi model pada test set
# -------------------------------------------------------
if "trained_model" in st.session_state:
    model = st.session_state["trained_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    st.markdown("<h3>2) Evaluasi Model</h3>", unsafe_allow_html=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    pr = average_precision_score(y_test, y_proba) if y_proba is not None else None
    auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("PR-AUC", f"{pr:.3f}" if pr is not None else "n/a")
    m3.metric("ROC AUC", f"{auc_roc:.3f}" if auc_roc is not None else "n/a")
    cl_counts = y_test.value_counts().to_dict()
    m4.metric("Test size", ", ".join([f"{k}: {v}" for k, v in cl_counts.items()]))

    st.markdown("### Interpretasi Metrics")
    st.markdown(
        f"- **Accuracy:** {acc:.3f} — persentase prediksi benar. "
        "Nilai > 0.8 umumnya sudah baik, < 0.7 perlu ditinjau ulang."
    )
    if pr is not None:
        st.markdown(
            f"- **PR-AUC (Average Precision):** {pr:.3f} — fokus pada kelas positif."
        )
    if auc_roc is not None:
        st.markdown(
            f"- **ROC-AUC:** {auc_roc:.3f} — semakin mendekati 1, "
            "semakin baik pemisahan kelas."
        )
    st.markdown(
        "- **Precision:** Seberapa banyak prediksi positif yang benar "
        "(sedikit false positive)."
    )
    st.markdown(
        "- **Recall:** Seberapa banyak kasus positif yang berhasil tertangkap "
        "(sedikit false negative)."
    )
    st.markdown(
        "- **Confusion matrix:** Menunjukkan jumlah prediksi benar/salah per kelas."
    )

    # -------- Classification report --------
    st.markdown("#### Classification Report")
    st.info(
        "**Classification Report**: Ringkasan precision, recall, F1-score, "
        "dan support (jumlah sampel) untuk tiap kelas."
    )
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(
        pd.DataFrame(cr).transpose().style.format("{:.3f}"),
        height=250,
    )

    # -------- Confusion matrix --------
    st.markdown("#### Confusion Matrix")
    st.info(
        "**Confusion Matrix**: Baris = label aktual, kolom = hasil prediksi. "
        "Diagonal besar → prediksi banyak yang benar."
    )
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = make_fig(3.5, 2.5)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    col_cm1, col_cm2, col_cm3 = st.columns([1, 2, 1])
    with col_cm2:
        st.pyplot(fig_cm, use_container_width=True)

    # -------- ROC Curve --------
    if y_proba is not None:
        st.markdown("#### ROC Curve")
        st.info(
            "**ROC Curve**: Menunjukkan trade-off antara True Positive Rate (TPR) "
            "dan False Positive Rate (FPR)."
        )
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = make_fig(3.5, 2.5)
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}", linewidth=2)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()

        col_roc1, col_roc2, col_roc3 = st.columns([1, 2, 1])
        with col_roc2:
            st.pyplot(fig_roc, use_container_width=True)

    # -------- Precision–Recall Curve --------
    if y_proba is not None:
        st.markdown("#### Precision-Recall Curve")
        st.info(
            "**Precision-Recall Curve**: Berguna untuk dataset tidak seimbang, "
            "memperlihatkan trade-off antara Precision dan Recall."
        )
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig_pr, ax_pr = make_fig(3.5, 2.5)
        ax_pr.step(recall, precision, where="post", linewidth=2)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"Precision-Recall (AP = {pr:.3f})")

        col_pr1, col_pr2, col_pr3 = st.columns([1, 2, 1])
        with col_pr2:
            st.pyplot(fig_pr, use_container_width=True)

    # -------- Feature importance --------
    st.markdown("#### Feature Importance")
    st.info(
        "Skor importance menunjukkan seberapa besar kontribusi masing-masing fitur "
        "dalam keputusan model."
    )

    try:
        # --- Ambil feature importance untuk tabel (seluruh fitur) ---
        feat_imp_all = pd.Series(
            model.feature_importances_,
            index=df.drop(columns=[target_col]).columns,
        ).sort_values(ascending=False)

        # --- Tabel (seluruh fitur) ---
        st.dataframe(
            feat_imp_all.reset_index()
            .rename(columns={"index": "feature", 0: "importance"})
            .style.format({"importance": "{:.4f}"}),
            height=300,
        )

        # --- Ambil Top 10 untuk visualisasi ---
        feat_imp_top10 = feat_imp_all.head(10)

        # --- Atur ukuran gambar ---
        fig_fi, ax_fi = make_fig(12,10)

        sns.barplot(
            x=feat_imp_top10.values,
            y=feat_imp_top10.index,
            ax=ax_fi,
            palette="viridis"
        )

        ax_fi.set_xlabel("Importance")
        ax_fi.set_ylabel("")
        ax_fi.set_title("Top 10 Feature Importance")

        col_fi1, col_fi2, col_fi3 = st.columns([1, 3, 1])
        with col_fi2:
            st.pyplot(fig_fi, use_container_width=True)

    except Exception:
        st.info("Feature importance tidak tersedia untuk model ini.")

# -------------------------------------------------------
# Analisis risiko per pasien (tabel skor risiko)
# -------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Distribusi Risiko Pasien")
st.markdown("<br>", unsafe_allow_html=True)

if "df_risk" in st.session_state and st.session_state["df_risk"] is not None:
    df_risk = st.session_state["df_risk"]

    low = int((df_risk["risk_level"] == "Rendah").sum())
    med = int((df_risk["risk_level"] == "Sedang").sum())
    high = int((df_risk["risk_level"] == "Tinggi").sum())
    tot = len(df_risk)

    k1, k2, k3 = st.columns(3)
    k1.markdown(
        f"""
        <div class="metric-card low">
          Rendah<br>
          <span style="font-size:24px">{low}</span>
          <div style="font-weight:400;font-size:13px">{low/tot*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""
        <div class="metric-card mid">
          Sedang<br>
          <span style="font-size:24px">{med}</span>
          <div style="font-weight:400;font-size:13px">{med/tot*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"""
        <div class="metric-card high">
          Tinggi<br>
          <span style="font-size:24px">{high}</span>
          <div style="font-weight:400;font-size:13px">{high/tot*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Tabel Risiko")
    st.info(
        "**Tabel Risiko**: Menampilkan setiap pasien dengan skor risiko (0–1) "
        "dan level (Rendah/Sedang/Tinggi). Cocok untuk prioritas tindak lanjut."
    )

    sort_order = st.radio(
        "Urutkan Risiko:",
        ["Tertinggi", "Terendah"],
        horizontal=True,
    )
    level_filter = st.multiselect(
        "Filter level (kosong = semua):",
        options=["Tinggi", "Sedang", "Rendah"],
        default=["Tinggi", "Sedang", "Rendah"],
    )

    df_show = df_risk[df_risk["risk_level"].isin(level_filter)]
    ascending = False if sort_order.startswith("Tertinggi") else True
    df_show = df_show.sort_values("risk_score", ascending=ascending).reset_index(
        drop=True
    )

    st.dataframe(df_show, height=420, use_container_width=True)

    # Tombol unduh tabel risiko
    csv_buf = df_show.to_csv(index=False).encode("utf-8")

    col1, col2, col3 = st.columns([1, 20, 1])
    with col2:
        st.download_button(
            "Unduh CSV Risiko",
            data=csv_buf,
            file_name="risk_table.csv",
            mime="text/csv",
        )

    # Tombol lanjut ke halaman visualisasi
    col1, col2, col3 = st.columns([1, 20, 1])
    with col2:
        if st.button("Lanjut ke Visualisasi", use_container_width=True):
            st.switch_page("pages/4_Visualisasi.py")
