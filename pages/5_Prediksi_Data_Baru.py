import streamlit as st
import pandas as pd
import numpy as np

# ============================= CSS =======================================
st.markdown("""
<style>
.main-title {
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
.form-card {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-top: 10px;
}
.form-title {
    font-size: 1.4rem;
    font-weight: bold;
    margin-bottom: 10px;
    color: #0d47a1;
}
.input-label {
    font-weight: 600;
    margin-bottom: 4px;
}
.stButton > button {
    font-weight: 700 !important;
    color: #0d47a1 !important;
    background: white !important;
    border: 2px solid #0d47a1 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(13,71,161,0.12) !important;
    padding: 14px 0 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    margin-top: 20px !important;
    margin-bottom: 20px !important;
}
.stButton > button:hover {
    background-color: #0d47a1 !important;
    color: #ffffff !important;
    box-shadow: 0 6px 16px rgba(13,71,161,0.18) !important;
}

</style>
""", unsafe_allow_html=True)

# =================== HEADER ==============================================
st.markdown("<div class='main-title'>Sistem Analisis Risiko Diabetes</div>", unsafe_allow_html=True)

# ================== DESKRIPSI ============================================
st.markdown("""
<div class='form-card'>
<h3>5. Prediksi Risiko Diabetes Untuk Data Baru</h3>
<p>Masukkan data baru dan model akan memprediksi risiko diabetes beserta probabilitasnya.</p>
</div>
""", unsafe_allow_html=True)

# ======================= VALIDASI SESSION ================================
required_keys = [
    "model",
    "processed_data",
    "prep_selected_cols",
    "original_selected_cols",
    "encoder_objects",
    "final_feature_names",
]

for key in required_keys:
    if key not in st.session_state:
        st.error(f"Data '{key}' belum tersedia. Lakukan preprocessing & training dulu.")
        st.stop()

model = st.session_state["model"]
df_processed = st.session_state["processed_data"]
original_selected_cols = st.session_state["original_selected_cols"]
encoder_objects = st.session_state["encoder_objects"]
final_feature_names = st.session_state["final_feature_names"]

# Buang target dari daftar fitur jika masih nyangkut
if "target_col" in st.session_state:
    target_col = st.session_state["target_col"]
    original_selected_cols = [c for c in original_selected_cols if c != target_col]
else:
    target_col = None

# Sembunyikan kolom non-fitur (id, index, dsb)
cols_for_form = [
    c for c in original_selected_cols
    if c.lower() not in ("id", "no", "index")
]

# Referensi data sebelum encoding bila ada
if "df_original" in st.session_state:
    df_ref = st.session_state["df_original"]
else:
    df_ref = df_processed  # fallback

if not isinstance(df_ref, pd.DataFrame):
    df_ref = pd.DataFrame(df_ref)

# ==================== DESKRIPSI FITUR (BISA DIEDIT) ======================
feature_desc = {
    "age": "Usia pasien (tahun).",
    "hypertension": "0 = tidak hipertensi, 1 = hipertensi.",
    "heart_disease": "0 = tidak ada riwayat penyakit jantung, 1 = ada.",
    "ever_married": "Status pernah menikah (Yes/No).",
    "Residence_type": "Tipe tempat tinggal (Urban/Rural).",
    "avg_glucose_level": "Rata-rata kadar glukosa darah.",
    "bmi": "Body Mass Index (kg/m²).",
    "gender": "Jenis kelamin pasien.",
    "work_type": "Tipe pekerjaan.",
    "smoking_status": "Status merokok.",
}

# ================== INFO ENCODER (label / ohe) ===========================
label_encoded_cols = [
    col for col, enc in encoder_objects.items() if enc.get("type") == "label"
]
ohe_encoded_cols = [
    col for col, enc in encoder_objects.items() if enc.get("type") == "ohe"
]

# list untuk menyimpan fitur numerik biner (0/1)
binary_numeric_cols = []

# ===================== PILIH CONTOH BARIS (OPSIONAL) =====================
st.markdown("<br><div class='form-title'>Form Input Data Baru</div>", unsafe_allow_html=True)

# Inisialisasi session state untuk sample index
if "sample_index_state" not in st.session_state:
    st.session_state["sample_index_state"] = 0

sample_row = None
if df_ref is not None and not df_ref.empty:
    st.markdown(
        "**Pilihan Cepat (opsional):** Gunakan nilai dari baris tertentu di dataset sebagai nilai awal."
    )
    use_sample = st.checkbox(
        "Gunakan contoh baris dari dataset sebagai nilai awal", value=False
    )
    
    if use_sample:
        max_index = len(df_ref) - 1
        sample_index = st.slider(
            "Pilih indeks baris contoh", 
            0, 
            max_index, 
            st.session_state["sample_index_state"],
            key="sample_slider"
        )
        # Update session state
        st.session_state["sample_index_state"] = sample_index
        sample_row = df_ref.iloc[sample_index]
        st.caption(f"Nilai awal diambil dari baris ke-{sample_index} pada dataset.")
        
        # Tampilkan preview data yang dipilih
        st.dataframe(sample_row.to_frame().T, use_container_width=True)

# ===================== FORM INPUT ========================================
input_values = {}

with st.form("form_prediksi"):
    st.write("Masukkan nilai untuk setiap fitur yang digunakan model:")

    cols_per_row = 2
    cols = st.columns(cols_per_row)

    for i, col in enumerate(cols_for_form):

        with cols[i % cols_per_row]:
            desc_label = feature_desc.get(col, col)
            st.markdown(
                f"<div class='input-label'>{desc_label}</div>",
                unsafe_allow_html=True,
            )

            is_label = col in label_encoded_cols
            is_ohe = col in ohe_encoded_cols
            has_ref_col = col in df_ref.columns

            # Nilai contoh bila user pilih sample row
            sample_val = None
            if sample_row is not None and has_ref_col and col in sample_row.index:
                sample_val = sample_row[col]

            # ---------- 1) LABEL ENCODER ----------
            if is_label:
                le = encoder_objects[col]["encoder"]
                classes = list(le.classes_)

                default_cat = classes[0]
                if sample_val is not None:
                    # Normalisasi sample_val sesuai preprocessing
                    normalized_val = str(sample_val).strip().lower()
                    if normalized_val in classes:
                        default_cat = normalized_val

                input_values[col] = st.selectbox(
                    f"Pilih {col}",
                    options=classes,
                    index=classes.index(default_cat),
                    key=f"label_{col}",
                )

            # ---------- 2) ONE-HOT ENCODER ----------
                        # ---------- 2) ONE-HOT ENCODER ----------
            elif is_ohe:
                ohe_cols = encoder_objects[col]["columns"]  # kolom dummy yang ada (drop_first)
                # 1) Prioritas: gunakan original_categories jika tersedia
                if "original_categories" in st.session_state and col in st.session_state["original_categories"]:
                    all_categories = st.session_state["original_categories"][col]
                else:
                    # fallback ke df_ref (tapi berikan info/debug)
                    if has_ref_col and col in df_ref.columns:
                        all_categories = sorted(
                            df_ref[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
                        )
                    else:
                        all_categories = []

                # dari ohe_cols ambil nama kategori yang tersisa (kolom dummy)
                categories_from_cols = []
                for name in ohe_cols:
                    if name.startswith(f"{col}_"):
                        cat = name.split(f"{col}_", 1)[1]
                        categories_from_cols.append(cat)

                # jika all_categories tersedia, tentukan reference (kategori yang di-drop)
                if all_categories:
                    reference_categories = [c for c in all_categories if c not in categories_from_cols]
                    # Jaga urutan: letakkan reference pertama sesuai all_categories urut asli
                    # Gabungkan reference (urut asli) lalu kolom dummy (urut lexicographic agar konsisten)
                    categories = reference_categories + categories_from_cols
                else:
                    categories = categories_from_cols

                # terakhir normalisasi dan fallback
                categories = [str(c).strip().lower() for c in categories] if categories else ["-"]

                default_cat = categories[0]
                if sample_val is not None:
                    normalized_val = str(sample_val).strip().lower()
                    if normalized_val in categories:
                        default_cat = normalized_val

                input_values[col] = st.selectbox(
                    f"Pilih {col}",
                    options=categories,
                    index=categories.index(default_cat),
                    key=f"ohe_{col}",
                )

            # ---------- 3) KATEGORIKAL BERDASAR DTYPE ----------
            elif has_ref_col and df_ref[col].dtype == object:
                col_data = df_ref[col]
                # Normalisasi nilai unik sesuai preprocessing
                unique_vals = sorted(
                    col_data.dropna().astype(str).str.strip().str.lower().unique().tolist()
                )

                if not unique_vals:
                    unique_vals = ["-"]

                default_cat = unique_vals[0]
                if sample_val is not None:
                    # Normalisasi sample_val
                    normalized_val = str(sample_val).strip().lower()
                    if normalized_val in unique_vals:
                        default_cat = normalized_val

                input_values[col] = st.selectbox(
                    f"Pilih {col}",
                    options=unique_vals,
                    index=unique_vals.index(default_cat),
                    key=f"cat_{col}",
                )

            # ---------- 4) NUMERIK ----------
            elif has_ref_col:
                col_data = df_ref[col]

                # deteksi otomatis numerik biner 0/1
                unique_vals = sorted(col_data.dropna().unique().tolist())
                is_binary_numeric = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})

                if is_binary_numeric:
                    binary_numeric_cols.append(col)
                    default_val = 0
                    if sample_val in (0, 1):
                        default_val = int(sample_val)

                    input_values[col] = st.selectbox(
                        f"Pilih nilai {col} (0 atau 1)",
                        options=[0, 1],
                        index=[0, 1].index(default_val),
                        key=f"bin_{col}",
                    )

                else:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    col_mean = col_data.mean()

                    if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                        col_min = float(col_data.fillna(0).min() if not col_data.dropna().empty else 0.0)
                        col_max = float(col_data.fillna(100).max() if not col_data.dropna().empty else 100.0)
                        if col_min == col_max:
                            col_max = col_min + 1.0

                    # Tentukan default
                    if sample_val is not None and not pd.isna(sample_val):
                        default_val = float(sample_val)
                    else:
                        default_val = float(col_mean) if not pd.isna(col_mean) else (col_min + col_max) / 2.0

                    default_val = max(min(default_val, col_max), col_min)

                    # Jika kolom age/umur/usia, pakai step=1 dan cast ke int
                    if col.lower() in ["age", "umur", "usia"]:
                        input_values[col] = st.number_input(
                            f"Masukkan nilai {col}",
                            min_value=int(col_min),
                            max_value=int(col_max),
                            value=int(round(default_val)),
                            step=1,
                            key=f"num_{col}",
                        )
                    else:
                        input_values[col] = st.number_input(
                            f"Masukkan nilai {col}",
                            min_value=float(col_min),
                            max_value=float(col_max),
                            value=float(default_val),
                            key=f"num_{col}",
                        )

            # ---------- 5) FALLBACK ----------
            else:
                input_values[col] = 0.0
                st.caption(
                    "(kolom ini tidak ditemukan pada data referensi, nilai diisi 0)"
                )

    submitted = st.form_submit_button(
        "Prediksi Risiko Diabetes", use_container_width=True
    )


# ===================== PREDIKSI ==========================================
if submitted:
    # Raw input → DataFrame dengan kolom asli
    X_raw = pd.DataFrame([input_values])[cols_for_form]

    # NORMALISASI INPUT KATEGORIKAL (sesuai preprocessing)
    for col in X_raw.columns:
        if X_raw[col].dtype == object or col in label_encoded_cols or col in ohe_encoded_cols:
            X_raw[col] = X_raw[col].astype(str).str.strip().str.lower()

    # ---- VALIDASI NILAI 0 / 1 UNTUK FITUR BINER ----
    for col in binary_numeric_cols:
        val = X_raw[col].iloc[0]
        if val not in (0, 1):
            st.error(
                f"Kolom **{col}** hanya boleh bernilai 0 atau 1. "
                "Perbaiki input terlebih dahulu."
            )
            st.stop()

    X_new = X_raw.copy()

    # Terapkan encoding sesuai PREPROCESS
    for col, enc in encoder_objects.items():

        # LABEL ENCODING
        if enc["type"] == "label":
            le = enc["encoder"]
            if col in X_new.columns:
                val = X_raw[col].iloc[0]
                if val not in le.classes_:
                    st.error(
                        f"Nilai '{val}' pada kolom '{col}' tidak dikenali model.\n"
                        f"Pilih salah satu dari: {list(le.classes_)}"
                    )
                    st.stop()
                X_new[col] = le.transform([val])[0]

        # ONE HOT ENCODING
        elif enc["type"] == "ohe":
            ohe_cols = enc["columns"]

            # inisialisasi semua kolom OHE = 0
            for oc in ohe_cols:
                X_new[oc] = 0

            val = X_raw[col].iloc[0] if col in X_raw.columns else None

            # Untuk semua kolom OHE
            col_name = f"{col}_{val}"
            if val is not None and col_name in ohe_cols:
                X_new[col_name] = 1

            if col in X_new.columns:
                X_new = X_new.drop(columns=[col])

    # Samakan kolom dengan yang dipakai saat training
    X_new = X_new.reindex(columns=final_feature_names, fill_value=0)

    # --- Prediksi ---
    proba_all = model.predict_proba(X_new)[0]
    prob_0 = float(proba_all[0])   # Tidak diabetes
    prob_1 = float(proba_all[1])   # Diabetes

    pred_class = int(model.predict(X_new)[0])

    # Tentukan kategori risiko berdasar prob_1
    if prob_1 < 0.30:
        risk_label = "Rendah"
        color = "green"
    elif prob_1 < 0.70:
        risk_label = "Sedang"
        color = "orange"
    else:
        risk_label = "Tinggi"
        color = "red"

    # ===================== TAMPILKAN HASIL ================================
    st.subheader("Hasil Prediksi")

    if pred_class == 1:
        st.error("**Klasifikasi: PASIEN DIABETES**")
    else:
        st.success("**Klasifikasi: PASIEN TIDAK DIABETES**")

    st.markdown(f"<p style='font-size:1rem;margin:5px 0;'>Probabilitas Tidak Diabetes: <b>{prob_0*100:.2f}%</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1rem;margin:5px 0;'>Probabilitas Diabetes: <b>{prob_1*100:.2f}%</b></p>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style='background:{color};color:white;text-align:center;padding:15px;
        border-radius:10px;font-size:18px;margin-top:10px;'>
        <b>Kategori Risiko: {risk_label}</b></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    "<div style='margin-top:20px;'></div>", 
    unsafe_allow_html=True
    )
    st.markdown("#### Ringkasan Data yang Diprediksi")
    st.dataframe(X_raw, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
