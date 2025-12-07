import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# Konfigurasi halaman
# -------------------------------------------------------
st.set_page_config(
    page_title="EDA & Preprocessing Diabetes",
    layout="wide"
)

# -------------------------------------------------------
# Header + CSS global
# -------------------------------------------------------
st.markdown(
    """
    <div style="
        background-color: #0d47a1;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    ">
      Sistem Analisis Risiko Diabetes
    </div>

    <style>
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .scroll-container {
        height: 500px;          /* tinggi container */
        overflow-y: scroll;     /* scroll vertikal */
        overflow-x: scroll;     /* scroll horizontal */
        border: 1px solid #ccc;
        padding: 10px;
    }

    /* TAB: tampilan lebih lebar dan rata tengah */
    .tab-container {
        display: flex;
        justify-content: center;
        margin: 30px 0;
        width: 100%;
    }

    [data-baseweb="tab-list"] {
        background-color: #0d47a1 !important;
        border-radius: 10px !important;
        padding: 8px !important;
        display: flex;
        justify-content: center;
        gap: 20px;
        width: 100%;
    }

    [data-baseweb="tab"] {
        color: white !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        margin: 0 !important;
        padding: 12px 24px !important;
        flex: 1;
        text-align: center;
    }

    .stForm .stButton button,
    .stButton button {
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

    .stForm .stButton button:hover,
    .stButton button:hover {
        background-color: #0d47a1 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(13,71,161,0.18);
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1e88e5 !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    [data-baseweb="tab"]:hover {
        background-color: #1565c0 !important;
        color: white !important;
    }

    /* Hilangkan border form bawaan */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Hilangkan padding atas besar */
    section.main > div {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Card pengantar EDA & Preprocessing
# -------------------------------------------------------
st.markdown(
    """
    <div class='card'>
      <h3>2. EDA Dan Preprocessing</h3>
      <p>
        Di halaman ini ditampilkan ringkasan dataset, distribusi target, fitur numerik
        dan kategorikal, korelasi antar fitur, serta langkah-langkah preprocessing.
      </p>
      <p>
        Gunakan informasi ini untuk memahami karakteristik data sebelum model
        dilatih pada halaman Analisis Data.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Ambil dataset dari session_state
# -------------------------------------------------------
if "dataset" not in st.session_state:
    st.warning("Dataset belum di-upload. Kembali ke menu Input Dataset terlebih dahulu.")
    st.stop()

df_original = st.session_state["dataset"].copy()

# -------------------------------------------------------
# Tabs: EDA & Preprocessing
# -------------------------------------------------------
st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
eda_tab, prep_tab = st.tabs(["EDA", "Preprocessing"])
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
#                           EDA
# ======================================================
with eda_tab:
    st.subheader("Exploratory Data Analysis (EDA)")

    # -------- 0. Preview dataset --------
    st.write("### Preview Dataset Asli")
    st.dataframe(df_original.head(), use_container_width=True)

    st.write("### Informasi Dataset")
    c1, c2, c3, c4 = st.columns(4)

    # jumlah baris, kolom, missing, duplikat dalam bentuk card
    c1.markdown(
        f'<div class="card"><div class="card-title">Jumlah Baris</div>'
        f'<div class="card-value">{df_original.shape[0]}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="card"><div class="card-title">Jumlah Kolom</div>'
        f'<div class="card-value">{df_original.shape[1]}</div></div>',
        unsafe_allow_html=True,
    )

    missing_total = int(df_original.isnull().sum().sum())
    dup_total = int(df_original.duplicated().sum())

    c3.markdown(
        f'<div class="card"><div class="card-title">Total Missing</div>'
        f'<div class="card-value">{missing_total}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="card"><div class="card-title">Total Duplikat</div>'
        f'<div class="card-value">{dup_total}</div></div>',
        unsafe_allow_html=True,
    )

    # -------- 1. Missing value --------
    st.write("### Missing Values")
    st.write(f"Total missing: **{missing_total}**")
    st.dataframe(df_original.isnull().sum(), use_container_width=True)

    if missing_total > 0:
        st.warning("Dataset memiliki missing values.")

    # -------- 2. Duplikasi --------
    st.write("### Duplikasi")
    st.write(f"Total duplikat: **{dup_total}**")
    if dup_total > 0:
        st.warning("Dataset memiliki data duplikat.")

    # -------- 3. Deteksi kolom target --------
    st.write("### Deteksi Kolom Target")

    # Jika sudah dipilih di halaman Input Dataset, gunakan langsung
    if (
        "target_col" in st.session_state
        and st.session_state.get("target_col") in df_original.columns
    ):
        target_col = st.session_state["target_col"]
        st.success(f"Kolom target diambil dari halaman Input Dataset: **{target_col}**")
    else:
        target_col = None
        for col in df_original.columns:
            if col.lower() in [
                "diabetes",
                "diagnosed_diabetes",
                "outcome",
                "diabetes_status",
                "class",
                "label",
                "target",
            ]:
                target_col = col
                break

        if target_col is None:
            st.error(
                "Kolom target tidak ditemukan. "
                "Silakan pilih kolom target di halaman Input Dataset terlebih dahulu."
            )
            st.stop()
        else:
            st.success(f"Kolom target terdeteksi otomatis: **{target_col}**")
            st.session_state["target_col"] = target_col

    # -------- 4. Statistik deskriptif fitur numerik --------
    st.write("### Statistik Deskriptif (Fitur Numerik)")

    num_cols = df_original.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols_no_target = [c for c in num_cols if c != target_col]

    if len(num_cols_no_target) == 0:
        st.info("Tidak ada fitur numerik selain kolom target.")
    else:
        st.dataframe(
            df_original[num_cols_no_target].describe(),
            use_container_width=True,
        )

    # -------- Boxplot fitur numerik --------
    if len(num_cols_no_target) > 0:
        st.write("### Boxplot Fitur Numerik")

        for c in num_cols_no_target:
            # Lewati kolom id
            if c.lower() == "id" or "id" in c.lower():
                continue

            # Lewati kolom yang isinya biner (0/1)
            try:
                unique_vals = df_original[c].dropna().unique()
                if len(unique_vals) <= 2:
                    continue
            except Exception:
                continue

            fig_bp, ax_bp = plt.subplots(figsize=(8, 4), dpi=120)
            sns.boxplot(x=df_original[c], ax=ax_bp)
            ax_bp.set_title(f"Boxplot - {c}", fontsize=9)
            plt.tight_layout()

            buf = BytesIO()
            fig_bp.savefig(buf, format="png")
            buf.seek(0)

            st.markdown(
                "<div style='display: flex; justify-content: center;'>",
                unsafe_allow_html=True,
            )
            st.image(buf, use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)

            plt.close(fig_bp)

    # -------- 5. Daftar kolom numerik & kategorikal --------
    st.write("### Kolom Numerik & Kategorikal")
    cat_cols = df_original.select_dtypes(include=["object"]).columns.tolist()

    st.success(f"Fitur numerik: {num_cols_no_target}")
    st.info(f"Fitur kategorikal: {cat_cols}")

    # -------- 6. Distribusi target --------
    st.write("### Distribusi Target")

    class_counts = df_original[target_col].value_counts()
    st.dataframe(class_counts, use_container_width=True)

    fig1, ax1 = plt.subplots(figsize=(5, 5), dpi=120)
    ax1.pie(
        class_counts,
        labels=class_counts.index,
        autopct="%1.1f%%",
        textprops={"fontsize": 7},
    )
    ax1.set_title("Distribusi Kelas Target", fontsize=9)
    plt.tight_layout()

    buf = BytesIO()
    fig1.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    st.markdown(
        "<div style='display: flex; justify-content: center;'>",
        unsafe_allow_html=True,
    )
    st.image(buf, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    plt.close(fig1)

    # Cek imbalance
    ratio = class_counts.min() / class_counts.max()
    if ratio < 0.4:
        st.warning("Dataset tidak seimbang (imbalanced).")
    else:
        st.success("Distribusi kelas relatif seimbang.")

    # -------- 7. Korelasi fitur + target --------
    st.write("### Heatmap Korelasi Fitur & Target")

    corr_cols = [c for c in num_cols_no_target if "id" not in c.lower()]
    if target_col not in corr_cols:
        corr_cols.append(target_col)

    corr_df = df_original[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()

    st.info(
        """
        **Interpretasi nilai korelasi (absolut):**
        - 0.00 – 0.19 : Sangat lemah  
        - 0.20 – 0.39 : Lemah  
        - 0.40 – 0.59 : Sedang  
        - 0.60 – 0.79 : Kuat  
        - 0.80 – 1.00 : Sangat kuat
        """
    )

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8) , dpi=120)
    sns.heatmap(
        corr,
        annot=True, 
        annot_kws={"size": 7},
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.4,
        square=True,
        cbar=True,
        ax=ax_corr,
    )
    ax_corr.set_title("Heatmap Korelasi Antar Fitur & Target", fontsize=12)
    plt.tight_layout()

    st.pyplot(fig_corr)

    # -------- 8. Ranking korelasi terhadap target --------
    st.write("### Pengaruh Fitur Terhadap Target")

    if target_col in corr.columns:
        target_corr = corr[target_col].drop(labels=[target_col]).sort_values(
            ascending=False
        )

        def corr_cat(v: float) -> str:
            v = abs(v)
            if v <= 0.19:
                return "Sangat lemah"
            if v <= 0.39:
                return "Lemah"
            if v <= 0.59:
                return "Sedang"
            if v <= 0.79:
                return "Kuat"
            return "Sangat kuat"

        corr_table = pd.DataFrame(
            {"Fitur": target_corr.index, "Korelasi": target_corr.values}
        )
        corr_table["Kategori"] = corr_table["Korelasi"].apply(corr_cat)

        st.dataframe(corr_table, use_container_width=True)
    else:
        st.info("Target bukan numerik sehingga korelasi tidak dapat dihitung.")

# ======================================================
#                     PREPROCESSING
# ======================================================
with prep_tab:
    st.subheader("Preprocessing Data")

    # Nilai teks yang akan dianggap sebagai NA
    na_markers = ["N/A", "NA", "n/a", "na"]

    # ---------------------------------------------------
    # Tombol Reset preprocessing
    # ---------------------------------------------------
    if st.button("Reset preprocessing", use_container_width=True):
        base_cols = st.session_state.get("prep_selected_cols", df_original.columns)

        reset_df = df_original[base_cols].copy()
        try:
            reset_df.replace(na_markers, np.nan, inplace=True)
        except Exception:
            pass

        st.session_state["prep_df"] = reset_df
        st.session_state["processed_data"] = reset_df.copy()
        st.session_state["prep_state"] = {
            "missing_done": False,
            "duplicate_done": False,
            "encoding_done": False,
            "outlier_done": False,
        }
        st.session_state["missing_drop_choice"] = []

        if "handled_outlier_cols" in st.session_state:
            st.session_state.pop("handled_outlier_cols")

        st.success("Preprocessing berhasil di-reset.")

    # Mulai ulang df kerja dari df_original
    df = df_original.copy()
    try:
        df.replace(na_markers, np.nan, inplace=True)
    except Exception:
        pass

    # ---------------------------------------------------
    # Inisialisasi variabel di session_state
    # ---------------------------------------------------
    if "prep_state" not in st.session_state:
        st.session_state["prep_state"] = {
            "missing_done": False,
            "duplicate_done": False,
            "encoding_done": False,
            "outlier_done": False,
        }

    if "prep_df" not in st.session_state:
        st.session_state["prep_df"] = df.copy()
        st.session_state["processed_data"] = df.copy()

    if "prep_selected_cols" not in st.session_state:
        st.session_state["prep_selected_cols"] = df.columns.tolist()

    if "missing_drop_choice" not in st.session_state:
        st.session_state["missing_drop_choice"] = []

    prep_state = st.session_state["prep_state"]

    # ---------------------------------------------------
    # 1. Pilih kolom relevan untuk pemodelan
    # ---------------------------------------------------
    st.write("### 1. Pilih Kolom Relevan")

    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Pilih fitur untuk pemodelan:",
        all_cols,
        default=[c for c in st.session_state.get("prep_selected_cols", all_cols) if c in all_cols],
    )

    # Jika pilihan kolom berubah, reset beberapa status
    if selected_cols != st.session_state["prep_selected_cols"]:
        st.session_state["prep_selected_cols"] = selected_cols
        st.session_state["prep_df"] = df[selected_cols].copy()
        st.session_state["processed_data"] = st.session_state["prep_df"].copy()

        if "handled_outlier_cols" in st.session_state:
            st.session_state.pop("handled_outlier_cols")

        st.session_state["prep_state"] = {
            "missing_done": False,
            "duplicate_done": False,
            "encoding_done": False,
            "outlier_done": False,
        }
        prep_state = st.session_state["prep_state"]
        st.session_state["missing_drop_choice"] = []

    df = st.session_state["prep_df"].copy()

    st.write("Preview data terpilih:")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------------------------------------------
    # 2. Analisis dan penanganan missing value
    # ---------------------------------------------------
    st.write("### 2. Analisis Missing Value")

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_cols = missing_count[missing_count > 0].index.tolist()

    if len(missing_cols) == 0:
        st.success("Tidak ada missing value.")
        prep_state["missing_done"] = True
    else:
        mv_info = []
        for col in missing_cols:
            col_type = (
                "Numerik" if df[col].dtype in ["int64", "float64"] else "Kategorikal"
            )
            skew_val = df[col].skew() if col_type == "Numerik" else None

            if col_type == "Kategorikal":
                imput = "Mode"
            else:
                # Rekomendasi mean/median berdasarkan skewness
                if skew_val is None:
                    imput = "Mean"
                else:
                    if abs(skew_val) <= 0.5:
                        imput = "Mean"
                    elif abs(skew_val) <= 1:
                        imput = "Median"
                    else:
                        imput = "Median"

            mv_info.append(
                {
                    "Kolom": col,
                    "Missing Count": int(missing_count[col]),
                    "Missing %": round(missing_percent[col], 3),
                    "Tipe": col_type,
                    "Skewness": "-" if skew_val is None else round(skew_val, 3),
                    "Rekomendasi": imput,
                    "DropAllowed": missing_percent[col] < 5,
                }
            )

        mv_df = pd.DataFrame(mv_info)

        with st.form("missing_form"):
            st.dataframe(mv_df, use_container_width=True)

            drop_candidates = [x["Kolom"] for x in mv_info if x["DropAllowed"]]

            if drop_candidates:
                st.multiselect(
                    "Kolom yang boleh drop baris (<5% missing):",
                    drop_candidates,
                    key="missing_drop_choice",
                    disabled=prep_state["missing_done"],
                )
            else:
                st.info(
                    "Tidak ada kolom dengan missing <5%. "
                    "Semua kolom perlu imputasi."
                )

            st.info(
                """
                ▪ Kolom <5% → boleh drop baris  
                ▪ Kolom ≥5% → wajib imputasi  
                ▪ Numerik → Mean / Median (sesuai skewness)  
                ▪ Kategorikal → Mode  
                """
            )

            apply_missing = st.form_submit_button(
                "Terapkan Penanganan Missing Value",
                use_container_width=True,
                disabled=prep_state["missing_done"],
            )

        if prep_state["missing_done"] and not apply_missing:
            st.info("Missing value sudah ditangani. Gunakan reset jika ingin mengulang.")

        if apply_missing and not prep_state["missing_done"]:
            df_proc = df.copy()
            drop_choice = st.session_state["missing_drop_choice"]
            before = len(df_proc)

            # 1) Drop rows untuk kolom yang dipilih
            for item in mv_info:
                col = item["Kolom"]
                if col in drop_choice:
                    df_proc = df_proc[df_proc[col].notna()]

            # 2) Imputasi untuk kolom lainnya
            for item in mv_info:
                col = item["Kolom"]
                if col in drop_choice:
                    continue

                method = item["Rekomendasi"]
                if method == "Mean":
                    df_proc[col].fillna(df_proc[col].mean(), inplace=True)
                elif method == "Median":
                    df_proc[col].fillna(df_proc[col].median(), inplace=True)
                else:
                    df_proc[col].fillna(df_proc[col].mode()[0], inplace=True)

            after = len(df_proc)
            diff = before - after

            st.success(
                f"Missing value berhasil ditangani. "
                f"Baris: {before} → {after} "
                f"({'-' + str(diff) if diff > 0 else '0'})"
            )

            st.session_state["prep_df"] = df_proc.copy()
            st.session_state["processed_data"] = df_proc.copy()
            prep_state["missing_done"] = True

    # ---------------------------------------------------
    # 3. Penghapusan duplikat
    # ---------------------------------------------------
    st.write("### 3. Penghapusan Duplikat")

    df = st.session_state["prep_df"].copy()
    dup_count = df.duplicated().sum()
    st.write(f"Jumlah duplikat saat ini: **{dup_count} baris**")

    dup_disabled = (not prep_state["missing_done"]) or prep_state["duplicate_done"]

    if not prep_state["missing_done"]:
        st.info("Selesaikan penanganan missing value terlebih dahulu.")

    # --- Tampilkan tombol Hapus Duplikat hanya jika ada duplikat ---
    if dup_count > 0:
        if st.button(
            "Hapus Duplikat",
            use_container_width=True,
            disabled=dup_disabled,
        ):
            before_dup = len(df)
            df = df.drop_duplicates()
            after_dup = len(df)
            diff_dup = before_dup - after_dup

            st.success(
                f"Duplikat dihapus. Baris: {before_dup} → {after_dup} "
                f"({'-' + str(diff_dup) if diff_dup > 0 else '0'})"
            )

            st.session_state["prep_df"] = df.copy()
            st.session_state["processed_data"] = df.copy()
            prep_state["duplicate_done"] = True
    else:
        if prep_state["missing_done"]:
            prep_state["duplicate_done"] = True
            st.success("Tidak ada baris duplikat pada data saat ini.")

    # Simpan df_original jika belum ada
    if "df_original" not in st.session_state:
        st.session_state.df_original = df.copy()
    
    # Inisialisasi daftar kolom asli untuk prediksi
    if "original_selected_cols" not in st.session_state:
        st.session_state["original_selected_cols"] = st.session_state.get(
            "prep_selected_cols", df.columns.tolist()
        )

    # Inisialisasi encoder objects (kosong jika dataset murni numerik)
    if "encoder_objects" not in st.session_state:
        st.session_state["encoder_objects"] = {}

    # ---------------------------------------------------
    # 4. Encoding fitur kategorikal
    # ---------------------------------------------------    
    st.write("### 4. Encoding Fitur Kategorikal")
    df = st.session_state["prep_df"].copy()
    target_col = st.session_state["target_col"]
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]

    # =======================
    # Preprocessing kategori
    # =======================
    for col in cat_cols:
        # Ubah ke string, hapus spasi di awal/akhir, ubah ke lowercase, dan isi missing dengan 'unknown'
        df[col] = df[col].astype(str).str.strip().str.lower().fillna("unknown")

    encoder_objects = {}  # {col: {type, encoder, columns}}

    original_categories = {
    col: sorted(df[col].unique())
    for col in cat_cols
    }
    st.session_state["original_categories"] = original_categories

    if len(cat_cols) == 0:
        st.success("Tidak ada fitur kategorikal yang perlu di-encode.")
        prep_state["encoding_done"] = True
    else:
        enc_info = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            rec = "Label Encoding" if n_unique <= 2 else "One Hot Encoding"
            enc_info.append(
                {"Kolom": col, "Unique Categories": n_unique, "Rekomendasi": rec}
            )

        st.dataframe(pd.DataFrame(enc_info), use_container_width=True)

        enc_disabled = (not prep_state["duplicate_done"]) or prep_state["encoding_done"]

        if not prep_state["duplicate_done"]:
            st.info("Selesaikan Missing Value dan Duplikat terlebih dahulu.")

        if st.button(
            "Lakukan Encoding",
            use_container_width=True,
            disabled=enc_disabled,
        ):
            df_enc = df.copy()

            for col in cat_cols:
                # Label Encoding untuk kolom dengan ≤2 kategori
                if df_enc[col].nunique() <= 2:
                    le = LabelEncoder()
                    df_enc[col] = le.fit_transform(df_enc[col])

                    encoder_objects[col] = {
                        "type": "label",
                        "encoder": le,
                        "columns": [col],
                    }

                # One Hot Encoding untuk kolom dengan >2 kategori
                else:
                    dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=False)  
                    df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)

                    encoder_objects[col] = {
                        "type": "ohe",
                        "columns": list(dummies.columns),
                    }


            # Simpan hasil encoding
            st.session_state["prep_df"] = df_enc.copy()
            st.session_state["processed_data"] = df_enc.copy()
            st.session_state["encoder_objects"] = encoder_objects

            # Simpan nama kolom asli (sebelum encoding), tanpa target
            st.session_state["original_selected_cols"] = [
                c for c in df.columns if c != target_col
            ]

            # Simpan nama kolom akhir (setelah encoding), tanpa target
            st.session_state["final_feature_names"] = [
                c for c in df_enc.columns if c != target_col
            ]

            prep_state["encoding_done"] = True

            st.success("Encoding selesai.")
            st.write(f"Jumlah kolom setelah encoding: **{df_enc.shape[1]}**")
            st.dataframe(df_enc.head(), use_container_width=True)

    # ---------------------------------------------------
    # 5. Penanganan outlier (opsional)
    # ---------------------------------------------------
    st.write("### 5. Penanganan Outlier (Opsional)")

    df = st.session_state["prep_df"].copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    outlier_disabled = (not prep_state.get("encoding_done", False)) or prep_state.get(
        "outlier_done", False
    )

    if not prep_state.get("encoding_done", False):
        st.info("Selesaikan langkah encoding terlebih dahulu.")
    else:
        if len(num_cols) == 0:
            st.info("Tidak ada kolom numerik untuk penanganan outlier.")
        else:
            # Jika outlier sudah ditangani, tampilkan ringkasan saja
            if prep_state.get("outlier_done", False):
                handled = st.session_state.get("handled_outlier_cols", [])
                st.success("Outlier sudah ditangani pada kolom yang dipilih sebelumnya.")
                if handled:
                    st.write(f"Kolom yang telah diproses: {handled}")
                st.write("Preview dataset setelah penanganan outlier:")
                st.dataframe(st.session_state.get("prep_df", df).head(), use_container_width=True)
            else:
                # Deteksi kandidat kolom outlier (bukan id, bukan biner)
                outlier_candidate_cols = []
                for c in num_cols:
                    if c not in df.columns:
                        continue
                    if c.lower() == "id" or "id" in c.lower():
                        continue
                    try:
                        if df[c].dropna().nunique() <= 2:
                            continue
                    except Exception:
                        continue
                    outlier_candidate_cols.append(c)

                outlier_stats = []
                for c in outlier_candidate_cols:
                    Q1 = df[c].quantile(0.25)
                    Q3 = df[c].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    mask = (df[c] < lower) | (df[c] > upper)
                    outlier_stats.append(
                        {
                            "Kolom": c,
                            "Lower": round(lower, 6) if pd.notna(lower) else None,
                            "Upper": round(upper, 6) if pd.notna(upper) else None,
                            "Outlier Count": int(mask.sum()),
                        }
                    )

                outlier_df = pd.DataFrame(outlier_stats)

                st.write("Tabel deteksi outlier (metode IQR 1.5):")
                if outlier_df.empty:
                    st.info(
                        "Tidak ada kolom numerik yang memenuhi kriteria untuk "
                        "deteksi outlier."
                    )
                else:
                    st.dataframe(outlier_df, use_container_width=True)

                cols_with_outliers = outlier_df[
                    outlier_df["Outlier Count"] > 0
                ]["Kolom"].tolist()

                st.write("Pilih metode penanganan outlier:")
                method = st.radio(
                    "Metode:",
                    options=[
                        "Tidak",
                        "Hapus Rows",
                        "Winsorize (cap)",
                        "Transform (log1p)",
                    ],
                    index=0,
                )

                selected_outlier_cols = []
                if method != "Tidak":
                    if len(cols_with_outliers) == 0:
                        st.info("Tidak ditemukan outlier pada kolom numerik.")
                    else:
                        selected_outlier_cols = st.multiselect(
                            "Pilih kolom yang akan diproses:",
                            cols_with_outliers,
                            default=cols_with_outliers,
                        )

                    if st.button(
                        "Terapkan Penanganan Outlier",
                        use_container_width=True,
                        disabled=outlier_disabled,
                    ):
                        df_out = df.copy()
                        before_out = len(df_out)
                        removed_counts = {}

                        def compute_bounds(series: pd.Series):
                            Q1 = series.quantile(0.25)
                            Q3 = series.quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            return lower, upper

                        if method == "Hapus Rows":
                            mask_keep = pd.Series(True, index=df_out.index)
                            for c in selected_outlier_cols:
                                lower, upper = compute_bounds(df_out[c])
                                mask_keep &= (df_out[c] >= lower) & (df_out[c] <= upper)
                                removed_counts[c] = int(
                                    ((df_out[c] < lower) | (df_out[c] > upper)).sum()
                                )
                            df_out = df_out[mask_keep]

                        elif method == "Winsorize (cap)":
                            for c in selected_outlier_cols:
                                lower, upper = compute_bounds(df_out[c])
                                removed_counts[c] = int(
                                    ((df_out[c] < lower) | (df_out[c] > upper)).sum()
                                )
                                df_out[c] = df_out[c].clip(lower=lower, upper=upper)

                        elif method == "Transform (log1p)":
                            for c in selected_outlier_cols:
                                lower, upper = compute_bounds(df_out[c])
                                removed_counts[c] = int(
                                    ((df_out[c] < lower) | (df_out[c] > upper)).sum()
                                )
                                new_col = f"{c}_log1p"
                                df_out[new_col] = np.log1p(df_out[c].clip(lower=0))

                        after_out = len(df_out)
                        diff_out = before_out - after_out

                        st.session_state["prep_df"] = df_out.copy()
                        st.session_state["processed_data"] = df_out.copy()
                        st.session_state["handled_outlier_cols"] = selected_outlier_cols.copy()
                        st.session_state["handled_outlier_method"] = method
                        prep_state["outlier_done"] = True

                        st.success(
                            f"Penanganan outlier ({method}) diterapkan. "
                            f"Baris: {before_out} → {after_out} ({diff_out} baris hilang)"
                        )

                        if removed_counts:
                            removed_df = pd.DataFrame(
                                {
                                    "Kolom": list(removed_counts.keys()),
                                    "Count": list(removed_counts.values()),
                                }
                            )
                            st.write(
                                "Jumlah outlier yang terdeteksi/ditangani per kolom:"
                            )
                            st.table(removed_df)

                        st.write("Preview dataset setelah penanganan outlier:")
                        st.dataframe(df_out.head(), use_container_width=True)

    # ---------------------------------------------------
    # 6. Finalisasi preprocessing dan lanjut ke Analisis
    # ---------------------------------------------------
    st.write("### 6. Finalisasi Preprocessing")

    if prep_state.get("encoding_done", False):
        if "processed_data" not in st.session_state:
            st.session_state["processed_data"] = st.session_state.get(
                "prep_df", df
            ).copy()

        processed_df = st.session_state.get(
            "processed_data",
            st.session_state.get("prep_df", df),
        ).copy()

        rows, cols = processed_df.shape

        st.success("Preprocessing selesai. Ringkasan data akhir:")
        st.write(f"- Jumlah baris: **{rows}**")
        st.write(f"- Jumlah kolom: **{cols}**")

        st.write("Preview dataset akhir:")
        st.dataframe(processed_df.head(), use_container_width=True)

        if st.button("Lanjut ke Analisis Data", use_container_width=True):
            st.switch_page("pages/3_Analisis_Data.py")
    else:
        st.info(
            "Selesaikan semua tahapan preprocessing (termasuk encoding) "
            "untuk melanjutkan ke halaman Analisis Data."
        )

