# app.py
# Streamlit 1.11.0 compatible; no sklearn; UI only – logic lives in the modules.
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Import modul logika (dengan fallback aman) ---
try:
    import DataCleaning as DC
except Exception as e:
    DC = None

try:
    import DataIntegration as DI
except Exception as e:
    DI = None

try:
    import DataTransformation as DT
except Exception as e:
    DT = None

try:
    import DataReduction as DR
except Exception as e:
    DR = None

try:
    import DataDiscretization as DD
except Exception as e:
    DD = None

# --- Import modul Visualisasi (controller) ---
try:
    from Visualisasi import Visualisasi as VIS  # jika folder Visualisasi adalah package
except Exception:
    try:
        import Visualisasi.Visualisasi as VIS   # alternatif path
    except Exception:
        VIS = None



# ---------- Utils ----------
def line():
    st.markdown('---')

def init_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_work" not in st.session_state:
        st.session_state.df_work = None
    if "logs" not in st.session_state:
        st.session_state.logs = []

def log(msg):
    st.session_state.logs.append(str(msg))

def preview_df(df, title="Preview Data"):
    st.subheader(title)
    if df is None:
        st.info("Belum ada data.")
        return
    # kasih key unik berdasarkan judul
    key = f"slider_{title.replace(' ', '_')}"
    n_rows = st.slider("Tampilkan n baris awal", 5, 1000, 20, key=key)
    st.write(df.head(n_rows))
    st.caption(f"Shape: {df.shape[0]} baris × {df.shape[1]} kolom")


def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def cat_columns(df: pd.DataFrame):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ---------- App ----------
st.set_page_config(page_title="Preprocessing App", layout="wide")
init_state()

st.title("🧹 Preprocessing Data")

# Sidebar Navigation
st.sidebar.header("Navigasi")
page = st.sidebar.selectbox(
    "Tahap",
    [
        "1) Upload Data",
        "2) Data Cleaning",
        "3) Data Integration",
        "4) Data Transformation",
        "5) Data Reduction (PCA)",
        "6) Data Discretization",
        "7) Log & Unduh",
        "8) Visualisasi Clustering",
        "9) Insight Cluster",
        "10) Action Plan (Wisdom)",
    ],
)

# 1) Upload Data
if page == "1) Upload Data":
    st.subheader("Upload Dataset")
    st.write("Dukung: CSV, XLSX")
    file = st.file_uploader("Pilih file", type=["csv", "xlsx"])

    colA, colB = st.columns(2)
    with colA:
        if st.button("Contoh Dummy 1000 Responden"):
            # bikin dummy kecil agar cepat
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "Nama": [f"Responden_{i+1}" for i in range(1000)],
                "Jenis Kelamin": rng.choice(["Laki-laki", "Perempuan"], 1000),
                "Semester": rng.integers(1, 9, 1000),
                "Jurusan": rng.choice(["Informatika","Matematika","Fisika","Biologi","Kimia"], 1000),
                "Kualitas Pengajaran": rng.integers(1, 6, 1000),
                "Kurikulum Relevan": rng.integers(1, 6, 1000),
                "Motivasi Belajar": rng.integers(1, 6, 1000),
                "Beban Tugas Wajar": rng.integers(1, 6, 1000),
                "Kesempatan Akademik": rng.integers(1, 6, 1000),
                "Kondisi Finansial": rng.integers(1, 6, 1000),
                "Stres Finansial": rng.integers(1, 6, 1000),
                "Kesehatan Fisik": rng.integers(1, 6, 1000),
                "Tingkat Energi": rng.integers(1, 6, 1000),
                "Kesehatan Mental": rng.integers(1, 6, 1000),
                "Dukungan Sosial": rng.integers(1, 6, 1000),
                "Kualitas Relasi": rng.integers(1, 6, 1000),
            })
            # tanam missing ~5%
            for c in df.columns[4:]:
                idx = df.sample(frac=0.05, random_state=42).index
                df.loc[idx, c] = np.nan
            # tanam outlier ~1%
            for c in df.columns[4:]:
                idx = df.sample(frac=0.01, random_state=123).index
                df.loc[idx, c] = df[c].mean(skipna=True) + rng.integers(5, 20)
            st.session_state.df_raw = df.copy()
            st.session_state.df_work = df.copy()
            log("Muat dummy dataset 1000 responden.")
    with colB:
        if st.button("Reset"):
            st.session_state.df_raw = None
            st.session_state.df_work = None
            st.session_state.logs = []

    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        st.session_state.df_raw = df.copy()
        st.session_state.df_work = df.copy()
        log(f"Upload file: {file.name} – shape {df.shape}")

    preview_df(st.session_state.df_raw, "Data Awal")

# 2) Data Cleaning
elif page == "2) Data Cleaning":
    st.subheader("Data Cleaning")
    df = st.session_state.df_work
    preview_df(df, "Sebelum Cleaning")

    # === Reset khusus Cleaning ===
    st.markdown("**Reset Data Cleaning**")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("⏪ Reset ke Data Awal (df_raw)", key="btn_reset_cleaning"):
            if st.session_state.df_raw is not None:
                st.session_state.df_work = st.session_state.df_raw.copy()
                log("Reset Data Cleaning: df_work dikembalikan ke df_raw.")
                st.success("Berhasil reset: data kerja kembali ke data awal (df_raw).")
            else:
                st.warning("Belum ada df_raw. Upload data dulu di menu 'Upload Data'.")
    with col_reset2:
        if st.button("🧹 Bersihkan Log Cleaning (opsional)", key="btn_reset_cleaning_ui"):
            st.session_state.logs = [
                l for l in st.session_state.logs
                if not l.startswith("[Missing]") and not l.startswith("[Outlier]") and "Reset Data Cleaning" not in l
            ]
            st.info("Log cleaning dirapikan.")

    # === Hapus kolom tidak penting ===
    if df is not None:
        st.markdown("### 🧱 Hapus Kolom Tidak Penting")
        all_cols = df.columns.tolist()
        drop_cols = st.multiselect("Pilih kolom yang ingin dihapus:", all_cols, key="drop_cols_clean")
        if st.button("🗑️ Hapus Kolom yang Dipilih", key="btn_drop_cols"):
            st.session_state.df_work = df.drop(columns=drop_cols, errors="ignore")
            log(f"Hapus kolom: {drop_cols}")
            st.success(f"Kolom {drop_cols} berhasil dihapus.")

    # === Perbaiki kolom Timestamp (robust) ===
    if df is not None and "Timestamp" in df.columns:
        st.markdown("### ⏰ Perbaiki Format Timestamp")
        if st.button("🔧 Konversi Kolom 'Timestamp' ke Datetime", key="btn_fix_ts"):
            try:
                df_fixed = st.session_state.df_work.copy()
                s = df_fixed["Timestamp"]
                ts = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                if ts.isna().mean() > 0.5:
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().any():
                        best, best_notna = None, -1
                        for unit in ["s", "ms", "us", "ns"]:
                            cand = pd.to_datetime(num, unit=unit, errors="coerce")
                            cnt = cand.notna().sum()
                            if cnt > best_notna:
                                best, best_notna = cand, cnt
                        if best_notna > 0:
                            ts = best
                df_fixed["Timestamp"] = ts
                st.session_state.df_work = df_fixed
                log("Kolom Timestamp dikonversi ke datetime (robust).")
                st.success("Berhasil konversi kolom 'Timestamp'.")
            except Exception as e:
                st.error(f"Gagal konversi Timestamp: {e}")

    if df is not None:
        line()

        # === Missing Value ===
        st.markdown("**Missing Value**")
        strategy = st.selectbox(
            "Strategi penanganan missing",
            ["drop", "mean", "median", "mode", "constant"],
            key="missing_strategy",
        )
        const_val = None
        if strategy == "constant":
            const_val = st.text_input("Nilai konstanta (akan dicoba dikonversi numerik jika cocok)", "0", key="const_val")

        line()

        # === Outlier (Deteksi otomatis: Z-score) ===
        st.markdown("**Outlier (Deteksi otomatis: Z-score)**")
        outlier_mode = st.radio(
            "Mode penanganan",
            ["clip", "remove", "mark", "set_nan", "impute_median", "impute_constant"],
            index=2,  # default: mark
            key="outlier_mode",
        )
        z_thr = st.slider("Ambang |Z|", 2.0, 5.0, 3.0, 0.1, key="z_thr_auto")
        fill_value_out = None
        if outlier_mode == "impute_constant":
            fill_value_out = st.text_input("Nilai konstanta untuk imputasi outlier", "0", key="const_outlier")

        cols_num = numeric_columns(df) if df is not None else []
        target_cols = st.multiselect(
            "Kolom numerik untuk proses (kosongkan = semua numerik)",
            cols_num, key="target_cols_z"
        )

        # === Tombol Jalankan Cleaning ===
        if st.button("Jalankan Cleaning", key="btn_run_clean"):
            try:
                if DC and hasattr(DC, "clean_data"):
                    df_clean, clean_log = DC.clean_data(
                        df,
                        missing_method=strategy,
                        fill_value=const_val,
                        outlier_method="zscore",         # paksa deteksi Z-score otomatis
                        z_threshold=z_thr,
                        outlier_mode=outlier_mode,
                        fill_value_outlier=fill_value_out,
                        target_cols=target_cols if target_cols else None,
                    )
                    st.session_state.df_work = df_clean
                    log(clean_log if clean_log else "Cleaning selesai via DataCleaning.clean_data (Z-score).")
                else:
                    # ====== Fallback minimalis (tanpa modul DC) ======
                    work = df.copy()

                    # --- Normalisasi NaN & koersi numerik-like ---
                    def _normalize_nans_local(dfin):
                        return dfin.replace(
                            to_replace=[r'^\s*$', r'^\-$', r'^(na|n/a|NA|N/A|null|NULL)$'],
                            value=np.nan, regex=True
                        )
                    def _coerce_numeric_like_local(dfin, thresh=0.7):
                        out = dfin.copy()
                        for c in out.columns:
                            if pd.api.types.is_numeric_dtype(out[c]):
                                continue
                            parsed = pd.to_numeric(out[c], errors="coerce")
                            if parsed.notna().mean() >= thresh:
                                out[c] = parsed
                        return out

                    work = _normalize_nans_local(work)
                    work = _coerce_numeric_like_local(work)

                    # --- Missing handling ---
                    if strategy == "drop":
                        work = work.dropna()
                    elif strategy == "mean":
                        means = work.mean(numeric_only=True)
                        for c in means.index:
                            work[c] = work[c].fillna(means[c])
                        cat_cols_local = [c for c in work.columns if c not in means.index]
                        for c in cat_cols_local:
                            try:
                                m = work[c].mode(dropna=True).iloc[0]
                            except Exception:
                                m = "Unknown"
                            work[c] = work[c].fillna(m)
                    elif strategy == "median":
                        meds = work.median(numeric_only=True)
                        for c in meds.index:
                            work[c] = work[c].fillna(meds[c])
                        cat_cols_local = [c for c in work.columns if c not in meds.index]
                        for c in cat_cols_local:
                            try:
                                m = work[c].mode(dropna=True).iloc[0]
                            except Exception:
                                m = "Unknown"
                            work[c] = work[c].fillna(m)
                    elif strategy == "mode":
                        try:
                            work = work.fillna(work.mode().iloc[0])
                        except Exception:
                            for c in work.columns:
                                m = work[c].mode()
                                if not m.empty:
                                    work[c] = work[c].fillna(m.iloc[0])
                    elif strategy == "constant":
                        try:
                            v = pd.to_numeric(pd.Series([const_val]), errors="coerce").iloc[0]
                            v = v if pd.notna(v) else const_val
                        except Exception:
                            v = const_val
                        work = work.fillna(v)

                    # --- Outlier detection: Z-score (auto) + action ---
                    cols_apply = target_cols if target_cols else [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                    union_mask = pd.Series(False, index=work.index)
                    counts = {}

                    for c in cols_apply:
                        col = pd.to_numeric(work[c], errors="coerce")
                        mu, sd = col.mean(), col.std(ddof=0)
                        if sd == 0 or pd.isna(sd):
                            counts[c] = 0
                            if outlier_mode == "mark":
                                work[c + "_is_outlier"] = False
                            continue
                        z = (col - mu) / sd
                        mask = z.abs() > z_thr
                        counts[c] = int(mask.sum())
                        lo, hi = mu - z_thr * sd, mu + z_thr * sd

                        if outlier_mode == "clip":
                            work[c] = col.clip(lo, hi)
                        elif outlier_mode == "remove":
                            union_mask = union_mask | mask
                        elif outlier_mode == "mark":
                            work[c + "_is_outlier"] = mask
                        elif outlier_mode == "set_nan":
                            work.loc[mask, c] = np.nan
                        elif outlier_mode == "impute_median":
                            work.loc[mask, c] = np.nan
                            med = work[c].median(skipna=True)
                            work[c] = work[c].fillna(med)
                        elif outlier_mode == "impute_constant":
                            work.loc[mask, c] = np.nan
                            try:
                                k = pd.to_numeric(pd.Series([fill_value_out]), errors="coerce").iloc[0]
                                k = k if pd.notna(k) else fill_value_out
                            except Exception:
                                k = fill_value_out
                            work[c] = work[c].fillna(k)

                    if outlier_mode == "remove":
                        dropped = int(union_mask.sum())
                        work = work.loc[~union_mask].copy()
                        log(f"[Outlier][Fallback] remove: drop {dropped} baris")
                    else:
                        log(f"[Outlier][Fallback] mode={outlier_mode} | |Z|>{z_thr} | per kolom: {counts}")

                    st.session_state.df_work = work
                    log(f"[Missing][Fallback] strategi={strategy}")

                st.success("Cleaning selesai.")
            except Exception as e:
                st.error(f"Gagal cleaning: {e}")

        preview_df(st.session_state.df_work, "Sesudah Cleaning")


# 3) Data Integration — APPEND (stack baris)
elif page == "3) Data Integration":
    st.subheader("Data Integration — Append (gabung baris)")
    left_df = st.session_state.df_work
    preview_df(left_df, "Dataset Kiri (df_work)")

    st.markdown("### Upload Dataset Kanan (Right)")
    up_right = st.file_uploader(
        "Pilih file Right (CSV/XLSX)",
        type=["csv", "xlsx"],
        key="right_file_integration_append",
    )

    def _read_any(file_obj):
        if file_obj is None:
            return None
        name = file_obj.name.lower()
        try:
            return pd.read_csv(file_obj) if name.endswith(".csv") else pd.read_excel(file_obj)
        except Exception as e:
            st.error(f"Gagal membaca file Right: {e}")
            return None

    right_df = _read_any(up_right) if up_right else None
    if right_df is not None:
        preview_df(right_df, "Dataset Kanan (Right)")

    # Opsi append
    st.markdown("**Pengaturan Append**")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        mode = st.selectbox("Mode kolom", ["union (semua kolom)", "intersection (hanya kolom sama)"], index=0, key="append_mode")
    with col2:
        add_source = st.checkbox("Tambahkan kolom _source", value=True, key="append_add_source")
    with col3:
        left_name = st.text_input("Label sumber kiri", "A", key="append_left_name")
        right_name = st.text_input("Label sumber kanan", "B", key="append_right_name")

    mode_val = "union" if mode.startswith("union") else "intersection"

    if (left_df is not None) and (right_df is not None):
        if st.button("➕ Gabungkan (Append Baris)", key="btn_append_rows"):
            try:
                if DI and hasattr(DI, "append_rows"):
                    merged, logtxt = DI.append_rows(
                        left_df, right_df,
                        mode=mode_val,
                        add_source=add_source,
                        left_name=left_name,
                        right_name=right_name,
                    )
                else:
                    # fallback sederhana
                    if mode_val == "intersection":
                        common = [c for c in left_df.columns if c in right_df.columns]
                        L = left_df[common].copy(); R = right_df[common].copy()
                    else:
                        all_cols = sorted(list(set(left_df.columns) | set(right_df.columns)))
                        L = left_df.reindex(columns=all_cols).copy()
                        R = right_df.reindex(columns=all_cols).copy()
                    if add_source:
                        L["_source"] = left_name; R["_source"] = right_name
                    merged = pd.concat([L, R], axis=0, ignore_index=True, sort=False)
                    logtxt = (
                        f"[Fallback] Append mode={mode_val} → hasil {merged.shape[0]} baris × {merged.shape[1]} kolom."
                    )

                st.session_state.df_work = merged
                log(logtxt)
                st.success("Append selesai. Dataset kanan ditambahkan di bawah dataset kiri.")
            except Exception as e:
                st.error(f"Gagal append: {e}")

    preview_df(st.session_state.df_work, "Hasil Integrasi (df_work)")



# 4) Data Transformation (AUTO: mean per aspek + z-score per aspek)
elif page == "4) Data Transformation":
    st.subheader("Transformasi (Auto) — Skor Aspek + Z-Score")
    df = st.session_state.df_work
    preview_df(df, "Sebelum Transformasi (Raw)")

    # ---------- Helpers robust ----------
    def _make_unique_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        """Pastikan semua nama kolom unik: Kolom, Kolom.1, Kolom.2, ..."""
        seen = {}
        new_cols = []
        for c in df_in.columns:
            name = str(c)
            if name not in seen:
                seen[name] = 0
                new_cols.append(name)
            else:
                seen[name] += 1
                new_cols.append(f"{name}.{seen[name]}")
        out = df_in.copy()
        out.columns = new_cols
        return out

    def _as_series(df_in: pd.DataFrame, colname: str) -> pd.Series:
        """Selalu kembalikan 1D Series meskipun ada kolom duplikat."""
        obj = df_in[colname]
        if isinstance(obj, pd.DataFrame):
            # ambil kolom pertama kalau duplikat
            obj = obj.iloc[:, 0]
        return obj

    def _find_timestamp_col(df_in: pd.DataFrame):
        ex = [c for c in df_in.columns if str(c).strip().lower() == "timestamp"]
        if ex:
            return ex[0]
        fuzzy = [c for c in df_in.columns if "timestamp" in str(c).lower()]
        return fuzzy[0] if fuzzy else None

    def _fix_timestamp_if_any(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = _make_unique_columns(df_in.copy())
        ts_col = _find_timestamp_col(df_in)
        if not ts_col:
            return df_in
        s1d = _as_series(df_in, ts_col)
        s_num = pd.to_numeric(s1d, errors="coerce")
        if s_num.notna().sum() == 0:
            df_in[ts_col] = pd.to_datetime(s1d, errors="coerce")
            return df_in
        med = float(s_num.dropna().median())
        if med > 1e17:   unit = "ns"
        elif med > 1e12: unit = "ms"
        elif med > 1e9:  unit = "s"
        else:            unit = None
        df_in[ts_col] = pd.to_datetime(s_num, unit=unit, errors="coerce") if unit else pd.to_datetime(s1d, errors="coerce")
        return df_in

    def _z_series_local(x: pd.Series) -> pd.Series:
        mu = float(np.nanmean(x.values))
        sd = float(np.nanstd(x.values, ddof=0))
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sd

    def _detect_likert_columns_local(df_in, exclude_contains):
        LIKERT_VALUES = {1,2,3,4,5}
        cand = []
        for c in df_in.columns:
            s = pd.to_numeric(_as_series(df_in, c), errors="coerce")  # <-- selalu Series 1D
            nn = s.notna().sum()
            if nn == 0:
                continue
            si = s.dropna().round().astype(int)
            share = (si.isin(LIKERT_VALUES).sum()) / max(1, nn)
            if share >= 0.70 and si.nunique() <= 5 and all(ex.lower() not in str(c).lower() for ex in exclude_contains):
                cand.append(c)
        # pertahankan urutan asli
        return [c for c in df_in.columns if c in set(cand)]

    def _group_by_six_local(cols, domains):
        groups = {}; start = 0
        for d in domains:
            groups[d] = cols[start:start+6]
            start += 6
        return groups
    # ------------------------------------

    if df is not None:
        # Opsi diagnostik
        st.markdown("**Pengaturan Deteksi**")
        colA, colB, colC = st.columns(3)
        with colA:
            show_diag = st.checkbox("Tampilkan diagnosis deteksi Likert & grouping", value=True, key="diag_aspek_auto")
        with colB:
            ex_text = st.text_input(
                "Kecualikan kolom yang mengandung kata (pisahkan koma)",
                value="Semester,Timestamp,Nama,Jenis Kelamin,Jurusan",
                key="exclude_aspek_auto"
            )
            exclude_list = [s.strip() for s in ex_text.split(",") if s.strip()]
        with colC:
            force_internal = st.checkbox("Gunakan engine internal (abaikan modul DT)", value=True, key="force_internal_dt")

        st.caption(f"Duplikat kolom (sebelum proses): {df.columns[df.columns.duplicated()].tolist()}")

        st.markdown("**Proses Agregasi Otomatis**")
        if st.button("Hitung Skor Aspek + Z-Score (Auto)", key="btn_aspek_auto"):
            try:
                # 1) dedup + perbaiki timestamp
                base = _make_unique_columns(df.copy())
                df_fixed = _fix_timestamp_if_any(base)

                domains = ["Education", "Financial", "Physical", "Psychological", "Relational"]

                # 2) Pakai DT kalau diizinkan & ada; kalau gagal -> fallback
                use_dt = (not force_internal) and (DT is not None) and hasattr(DT, "aggregate_aspects_auto")
                df_new = None; agg_log = ""

                if use_dt:
                    try:
                        res = DT.aggregate_aspects_auto(
                            df_fixed, domains=domains, exclude_contains=exclude_list
                        )
                        # dukung return (df, log) atau (df, groups, log)
                        if isinstance(res, tuple) and len(res) == 2:
                            df_new, agg_log = res
                        elif isinstance(res, tuple) and len(res) == 3:
                            df_new, _, agg_log = res
                        else:
                            raise ValueError("Format return DT.aggregate_aspects_auto tidak dikenali")
                    except Exception as e_dt:
                        st.info(f"Modul DT gagal ({e_dt}). Menggunakan engine internal.")
                        use_dt = False  # jatuh ke fallback

                if not use_dt:
                    # --- FALLBACK INTERNAL ---
                    likert_cols = _detect_likert_columns_local(df_fixed, exclude_list)
                    groups = _group_by_six_local(likert_cols, domains)
                    work = df_fixed.copy()
                    logs = [f"[Deteksi Likert] total kandidat: {len(likert_cols)} kolom"]
                    # Mean per domain
                    for dname in domains:
                        cols = groups[dname]
                        if cols:
                            work[dname] = work[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                            logs.append(f"[Mean] {dname}: mean dari {cols}")
                        else:
                            work[dname] = np.nan
                            logs.append(f"[Mean] {dname}: kolom tidak cukup (butuh 6).")
                    # Z per domain (pastikan 1D)
                    for dname in domains:
                        ser = _as_series(work, dname)
                        work[f"{dname}_Z"] = _z_series_local(ser)
                        logs.append(f"[Z-score] {dname}_Z dibuat dari {dname}")
                    # Urutan: [Means...] + [Z-scores...]
                    mean_cols = domains
                    z_cols = [f"{d}_Z" for d in domains]
                    df_new = work[mean_cols + z_cols]
                    agg_log = "\n".join(logs)
                    # --- END FALLBACK ---

                st.session_state.df_work = df_new
                log(agg_log if agg_log else "Agregasi aspek (mean) + Z-score selesai.")
                st.success("Agregasi aspek (mean) + Z-score selesai.")

            except Exception as e:
                st.error(f"Gagal proses agregasi otomatis: {e}")

        # Diagnostik (opsional)
        if show_diag and st.session_state.df_work is not None:
            try:
                df_diag = _fix_timestamp_if_any(_make_unique_columns(st.session_state.df_work.copy()))
                likert_cols = _detect_likert_columns_local(df_diag, exclude_list)
                st.markdown("**Kolom Likert Terdeteksi (urut):**")
                st.write(likert_cols[:60])

                groups_diag = _group_by_six_local(likert_cols, domains)
                st.markdown("**Mapping Domain → 6 Item:**")
                for dname in domains:
                    st.write(f"- {dname}: {groups_diag[dname]}")
            except Exception as e:
                st.warning(f"Diagnosis deteksi gagal: {e}")
                
        preview_df(st.session_state.df_work, "Sesudah Transformasi (Skor & Z-Score)")


# 5) Data Reduction (PCA)
elif page.startswith("5) Data Reduction"):
    st.subheader("Data Reduction — PCA & UMAP")

    df = st.session_state.df_work
    preview_df(df, "Sebelum Reduksi")

    if df is None or len(df) == 0:
        st.info("Belum ada data. Upload/siapkan data terlebih dahulu.")
    else:
        # Pilih fitur numerik
        cols_num = numeric_columns(df)
        with st.expander("🔧 Pilih Kolom Fitur (numerik)", expanded=False):
            target_cols = st.multiselect(
                "Kolom yang akan direduksi (kosongkan = semua numerik)",
                cols_num, key="red_target_cols"
            )

        # Pilih metode
        method = st.radio(
            "Metode reduksi",
            ["PCA (SVD, tanpa sklearn)", "UMAP"],
            index=0, key="red_method"
        )

        keep_src = st.checkbox("Keep source features (jangan drop fitur asli)", value=True, key="red_keep_src")

        # Parameter
        if method.startswith("PCA"):
            colA, colB, colC = st.columns([1,1,1])
            with colA:
                n_comp = st.number_input("n_components", 1, 10, 2, 1, key="pca_ncomp")
            with colB:
                scale = st.checkbox("Z-score sebelum PCA", value=True, key="pca_scale")
            with colC:
                show_scree = st.checkbox("Hitung Scree Data", value=False, key="pca_scree")
        else:
            # UMAP
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                n_comp = st.number_input("n_components", 2, 5, 2, 1, key="umap_ncomp")
            with c2:
                n_neighbors = st.slider("n_neighbors", 5, 100, 15, 1, key="umap_neighbors")
            with c3:
                min_dist = st.slider("min_dist", 0.0, 0.99, 0.1, 0.01, key="umap_mindist")
            with c4:
                metric = st.selectbox("metric", ["euclidean","manhattan","cosine","hamming"], index=0, key="umap_metric")
            scale = st.checkbox("Z-score sebelum UMAP", value=True, key="umap_scale")
            rand_state = st.number_input("random_state", 0, 10_000, 42, 1, key="umap_rs")

        # Jalankan
        if st.button("🧭 Jalankan Reduksi", key="btn_run_reduction"):
            try:
                df_in = df[target_cols] if target_cols else df[cols_num]
                if df_in is None or df_in.shape[1] == 0:
                    st.warning("Tidak ada kolom numerik yang dipilih/tersedia.")
                else:
                    if method.startswith("PCA"):
                        if DR and hasattr(DR, "pca_reduce"):
                            df_red, model, red_log, evr = DR.pca_reduce(df_in, n_components=int(n_comp), scale_zscore=bool(scale))
                        else:
                            # Fallback lokal sederhana (gunakan DataReduction di memori jika tidak terimport)
                            from numpy.linalg import svd
                            X = df_in.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
                            if scale:
                                mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0]=1.0; X=(X-mu)/sd
                            U,S,VT = svd(X, full_matrices=False)
                            comps = VT[:int(n_comp)]
                            scores = X @ comps.T
                            cols = [f"PC{i+1}" for i in range(int(n_comp))]
                            df_red = pd.DataFrame(scores[:, :int(n_comp)], columns=cols, index=df.index)
                            evr = (S**2)/np.sum(S**2)
                            red_log = f"[PCA Fallback] OK: comp={n_comp}, Total EV={round(float(np.sum(evr[:int(n_comp)]))*100,2)}%."
                        # gabungkan hasil
                        if keep_src:
                            out = df.copy().join(df_red)
                        else:
                            # drop fitur input saja
                            out = df.drop(columns=df_in.columns, errors="ignore").join(df_red)
                        st.session_state.df_work = out
                        log(red_log)
                        st.success("PCA selesai. Komponen ditambahkan.")
                        # Scree data (opsional)
                        if method.startswith("PCA") and show_scree:
                            try:
                                if DR and hasattr(DR, "scree_data"):
                                    xs, ys = DR.scree_data(df_in, scale_zscore=bool(scale))
                                    st.write({"component": xs, "evr": ys[:10]})
                            except Exception:
                                pass

                    else:
                        # UMAP
                        if DR and hasattr(DR, "umap_reduce"):
                            df_red, model, red_log, _ = DR.umap_reduce(
                                df_in,
                                n_components=int(n_comp),
                                n_neighbors=int(n_neighbors),
                                min_dist=float(min_dist),
                                metric=str(metric),
                                scale_zscore=bool(scale),
                                random_state=int(rand_state),
                            )
                        else:
                            # Fallback: gunakan PCA tapi beri nama UMAP*
                            from numpy.linalg import svd
                            X = df_in.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
                            if scale:
                                mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0]=1.0; X=(X-mu)/sd
                            U,S,VT = svd(X, full_matrices=False)
                            comps = VT[:int(n_comp)]
                            scores = X @ comps.T
                            cols = [f"UMAP{i+1}" for i in range(int(n_comp))]
                            df_red = pd.DataFrame(scores[:, :int(n_comp)], columns=cols, index=df.index)
                            red_log = "[UMAP Fallback] 'umap-learn' tidak tersedia → pakai PCA dan dinamai UMAP*."
                        if keep_src:
                            out = df.copy().join(df_red)
                        else:
                            out = df.drop(columns=df_in.columns, errors="ignore").join(df_red)
                        st.session_state.df_work = out
                        log(red_log)
                        st.success("UMAP selesai. Embedding ditambahkan.")

            except Exception as e:
                st.error(f"Gagal reduksi: {e}")

    preview_df(st.session_state.df_work, "Sesudah Reduksi")


# 6) Data Discretization (Z-score only, anti-NaN)
elif page == "6) Data Discretization":
    st.subheader("Data Discretization — Basis Z-score (anti-NaN)")

    df = st.session_state.df_work
    preview_df(df, "Sebelum Discretization")

    aspects = ["Education", "Financial", "Physical", "Psychological", "Relational"]

    col1, col2 = st.columns(2)
    with col1:
        low_thr = st.number_input("Ambang bawah Z (Rendah)", value=-0.5, step=0.1, key="disc_z_low")
    with col2:
        high_thr = st.number_input("Ambang atas Z (Tinggi)", value=0.5, step=0.1, key="disc_z_high")

    if st.button("🔖 Buat Label (Z-score)", key="btn_disc_z"):
        try:
            if DD and hasattr(DD, "discretize_pipeline_z"):
                df_new, dlog = DD.discretize_pipeline_z(
                    df,
                    aspects=aspects,
                    low_thr=low_thr,
                    high_thr=high_thr,
                )
            else:
                # ---- Fallback lokal (tanpa modul DD) ----
                def _num(s): return pd.to_numeric(s, errors="coerce")
                def _find_items(dfin, a): return [c for c in dfin.columns if str(c).startswith(f"{a}_")]
                def _ensure_aspect_means_local(dfin, aspects_):
                    out = dfin.copy()
                    for a in aspects_:
                        if a not in out.columns:
                            items = _find_items(out, a)
                            if len(items) >= 2:
                                out[a] = out[items].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                            else:
                                out[a] = np.nan
                    return out
                def _ensure_aspect_z_local(dfin, aspects_):
                    out = _ensure_aspect_means_local(dfin, aspects_).copy()
                    for a in aspects_:
                        zc = f"{a}_Z"
                        if zc not in out.columns:
                            x = _num(out[a])
                            mu = float(np.nanmean(x))
                            sd = float(np.nanstd(x, ddof=0))
                            x_filled = x.fillna(mu if np.isfinite(mu) else 0.0)
                            if not np.isfinite(sd) or sd == 0:
                                out[zc] = 0.0
                            else:
                                out[zc] = (x_filled - (mu if np.isfinite(mu) else 0.0)) / sd
                    return out
                def _lab(v, lo, hi):
                    try:
                        x = float(v)
                    except Exception:
                        return "Menengah"
                    if not np.isfinite(x): return "Menengah"
                    if x <= lo:  return "Tidak Sejahtera"
                    if x >= hi:  return "Sejahtera"
                    return "Menengah"

                work = _ensure_aspect_z_local(df.copy(), aspects)
                for a in aspects:
                    work[f"Label_{a}"] = work[f"{a}_Z"].apply(lambda v: _lab(v, low_thr, high_thr))
                zcols = [f"{a}_Z" for a in aspects]
                Z = work[zcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                work["WB_Index_Z"] = Z.mean(axis=1)
                work["WB_Label"] = work["WB_Index_Z"].apply(lambda v: _lab(v, low_thr, high_thr))
                df_new, dlog = work, "[Fallback] Discretization Z selesai (anti-NaN)."

            st.session_state.df_work = df_new
            log(dlog)
            st.success("Discretization (Z) selesai. Label per-aspek & WB_Label ditambahkan (tanpa NaN).")
        except Exception as e:
            st.error(f"Gagal membuat label: {e}")

    # Ringkasan label (cek cepat tidak NaN)
    if st.session_state.df_work is not None:
        with st.expander("📊 Ringkasan Label", expanded=True):
            for c in [
                "WB_Label",
                "Label_Education","Label_Financial","Label_Physical","Label_Psychological","Label_Relational"
            ]:
                if c in st.session_state.df_work.columns:
                    st.write(f"**{c}**")
                    st.write(st.session_state.df_work[c].value_counts(dropna=False))

    preview_df(st.session_state.df_work, "Sesudah Discretization (Z)")




# 7) Log & Unduh
elif page == "7) Log & Unduh":
    st.subheader("Ringkasan Log")
    if len(st.session_state.logs) == 0:
        st.info("Belum ada log proses.")
    else:
        for i, l in enumerate(st.session_state.logs, 1):
            st.write(f"{i}. {l}")

    line()
    st.subheader("Unduh Dataset Hasil")
    df = st.session_state.df_work
    if df is None:
        st.info("Tidak ada data untuk diunduh.")
    else:
        # CSV buffer
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh CSV",
            data=csv,
            file_name="preprocessed.csv",
            mime="text/csv",
        )

# 8) Visualisasi Clustering
elif page == "8) Visualisasi Clustering":
    st.subheader("Visualisasi Clustering (K-Means / DBSCAN / Hierarchical)")

    df = st.session_state.df_work
    preview_df(df, "Dataset Saat Ini")

    if VIS is None:
        st.error("Modul Visualisasi belum tersedia/terimport. Pastikan folder 'Visualisasi' berisi file Visualisasi.py dan modul algoritmanya.")
    else:
        try:
            df_new, vis_log = VIS.render(df)
            # Jika df_new berbeda (mis. user menempel label), simpan kembali
            if df is not None and df_new is not None and not df_new.equals(df):
                st.session_state.df_work = df_new
                st.success("Perubahan (label) telah diterapkan ke df_work.")
            # Catat log
            if vis_log:
                log(vis_log)
        except Exception as e:
            st.error(f"Gagal menjalankan Visualisasi: {e}")

    preview_df(st.session_state.df_work, "Setelah Visualisasi (jika label ditempel)")

# 9) Insight Cluster — baca rata-rata aspek per cluster dan beri ringkasan
elif page == "9) Insight Cluster":
    st.subheader("🧠 Insight dari Hasil Cluster (berbasis Z-score direkomendasikan)")

    df = st.session_state.df_work
    preview_df(df, "Dataset Aktif (df_work)")

    if df is None or len(df) == 0:
        st.info("Belum ada data. Upload/siapkan data terlebih dahulu.")
    else:
        # --- Cari kolom label cluster ---
        label_candidates = [c for c in df.columns if c.lower().endswith("_label") or c.lower().startswith("cluster")]
        for c in ["KMeans_Label","DBSCAN_Label","Hierarchical_Label"]:
            if c in df.columns and c not in label_candidates:
                label_candidates.append(c)
        if not label_candidates:
            st.warning("Belum ada kolom label cluster. Jalankan K-Means/DBSCAN/Hierarchical dulu di halaman Visualisasi.")
        else:
            colA, colB = st.columns([1,1])
            with colA:
                label_col = st.selectbox("Pilih kolom label cluster", label_candidates, index=0, key="ins_label_col")
            with colB:
                basis = st.radio("Basis interpretasi", ["Z-score (direkomendasikan)", "Mean (1–5)"], index=0, key="ins_basis")

            base_aspects = ["Education","Financial","Physical","Psychological","Relational"]
            use_z = basis.startswith("Z-score")

            if use_z and all((a + "_Z") in df.columns for a in base_aspects):
                aspect_cols = [a + "_Z" for a in base_aspects]
                pretty = {"Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                          "Psychological_Z":"Psikologis","Relational_Z":"Relasional"}
                lo_default, hi_default = -0.5, 0.5
            else:
                # fallback pakai mean 1–5 bila kolom *_Z belum ada
                aspect_cols = [a for a in base_aspects if a in df.columns]
                pretty = {"Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                          "Psychological":"Psikologis","Relational":"Relasional"}
                lo_default, hi_default = 2.5, 3.5

            if len(aspect_cols) < 3:
                st.warning(f"Kolom aspek kurang. Ditemukan: {aspect_cols}. "
                           f"Pastikan sudah menjalankan Data Transformation atau menyediakan kolom *_Z.")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    low_thr = st.number_input("Ambang rendah", value=float(lo_default), step=0.1, key="ins_thr_low")
                with c2:
                    high_thr = st.number_input("Ambang tinggi", value=float(hi_default), step=0.1, key="ins_thr_high")
                with c3:
                    show_var = st.checkbox("Tampilkan varians dalam-cluster", value=False, key="ins_show_var")

                # --- Coba pakai modul helper bila tersedia ---
                try:
                    # prefer modul di Visualisasi/ClusterInsight.py
                    try:
                        from Visualisasi.ClusterInsight import cluster_profiles, label_clusters, cluster_variability
                        use_helper = True
                    except Exception:
                        use_helper = False

                    if use_helper:
                        prof, mu, sd = cluster_profiles(df, label_col, aspect_cols, use_zscore=use_z)
                        labels_tbl = label_clusters(prof, low_thr=low_thr, high_thr=high_thr, pretty_names=pretty)

                        from Visualisasi.ClusterInsight import cluster_counts, make_text_summaries

                        # Hitung jumlah anggota per cluster (urut sesuai index prof/labels_tbl)
                        order = labels_tbl.index.astype(str).tolist()
                        counts = cluster_counts(df, label_col, order=order)

                        # Nama basis untuk ditampilkan
                        basis_name = "Z-score" if use_z else "Mean (1–5)"

                        # Mapping pretty yang sama dengan di atas
                        pretty_map = {
                            "Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                            "Psychological_Z":"Psikologis","Relational_Z":"Relasional",
                            "Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                            "Psychological":"Psikologis","Relational":"Relasional",
                        }

                        summaries = make_text_summaries(
                            prof, labels_tbl, counts,
                            pretty_names=pretty_map,
                            basis_name=basis_name
                        )

                        st.markdown("### 📝 Ringkasan Naratif per Cluster")
                        for item in summaries:
                            with st.expander(item["title"], expanded=False):
                                st.markdown(item["body"])

                    else:
                        # ---- Fallback internal singkat tanpa modul ----
                        def _num(dfin: pd.DataFrame) -> pd.DataFrame:
                            out = pd.DataFrame(index=dfin.index)
                            for c in dfin.columns:
                                out[c] = pd.to_numeric(dfin[c], errors="coerce")
                            return out

                        X = _num(df[aspect_cols])
                        labs = df[label_col].astype(str)
                        data = X.join(labs.rename("_lab_")).dropna(axis=0, how="any")
                        # jika basis Z: asumsikan sudah Z; kalau mean, tidak diubah
                        prof = data.groupby("_lab_")[aspect_cols].mean().sort_index()

                        def _bucket(v, lo, hi):
                            try:
                                x = float(v)
                            except Exception:
                                return "Menengah"
                            if x >= hi: return "Sejahtera"
                            if x <= lo: return "Tidak Sejahtera"
                            return "Menengah"

                        cat = prof.applymap(lambda v: _bucket(v, low_thr, high_thr))
                        def _sumrow(row):
                            pos = [pretty.get(k, k) for k, v in row.items() if v == "Sejahtera"]
                            neg = [pretty.get(k, k) for k, v in row.items() if v == "Tidak Sejahtera"]
                            mid = [pretty.get(k, k) for k, v in row.items() if v == "Menengah"]
                            parts = []
                            if pos: parts.append(" & ".join(pos) + " Sejahtera")
                            if neg: parts.append(" & ".join(neg) + " Rendah")
                            if not parts and mid: parts.append("Mayoritas Menengah")
                            return " | ".join(parts) if parts else "Campuran"
                        labels_tbl = prof.copy()
                        labels_tbl["__Ringkasan__"] = cat.apply(_sumrow, axis=1)
                        for a in aspect_cols:
                            labels_tbl[f"Label_{a}"] = cat[a]

                    st.markdown("### Profil & Ringkasan per Cluster")
                    st.dataframe(labels_tbl.round(3))

                    # Unduh CSV ringkasan
                    st.download_button(
                        "💾 Unduh Insight Cluster (CSV)",
                        data=labels_tbl.to_csv(index=True).encode("utf-8"),
                        file_name="cluster_insight.csv",
                        mime="text/csv",
                    )

                    # Varians dalam-cluster (opsional)
                    if show_var:
                        if use_helper:
                            var_tbl = cluster_variability(df, label_col, aspect_cols)
                        else:
                            # fallback varians
                            def _num(dfin): return dfin.apply(pd.to_numeric, errors="coerce")
                            X = _num(df[aspect_cols])
                            labs = df[label_col].astype(str)
                            data = X.join(labs.rename("_lab_")).dropna(axis=0, how="any")
                            var_tbl = data.groupby("_lab_")[aspect_cols].var(ddof=0).join(
                                data.groupby("_lab_").size().rename("n")
                            )
                        st.markdown("### Varians Dalam-Cluster (lebih kecil = lebih kompak)")
                        st.dataframe(var_tbl.round(3))

                    # Opsional: injeksi label manusiawi ke df_work
                    if st.checkbox("Tambahkan nama cluster (ringkasan) ke dataset", value=False, key="ins_add_human"):
                        df_new = df.copy()
                        name_map = labels_tbl["__Ringkasan__"].to_dict()
                        df_new["Cluster_Label_Human"] = df_new[label_col].astype(str).map(name_map)
                        st.session_state.df_work = df_new
                        st.success("Kolom Cluster_Label_Human ditambahkan ke df_work.")

                    st.caption("Catatan: basis Z-score membuat antar-aspek komparabel. Ubah ambang untuk sensitivitas label. "
                               "Untuk DBSCAN, label -1 (noise) akan dihitung sebagai satu kelompok bila ada.")

                except Exception as e:
                    st.error(f"Gagal membangun insight cluster: {e}")

# 10) Action Plan (Wisdom)
elif page == "10) Action Plan (Wisdom)":
    st.subheader("🧭 Rencana Aksi (Wisdom) — dari Insight ke Kebijakan")

    # Import tool dari Insight & Wisdom
    try:
        from Visualisasi.ClusterInsight import (
            cluster_profiles, label_clusters, cluster_counts, make_policy_recommendations
        )
        from Visualisasi.ClusterWisdom import (
            build_action_plan, plan_to_markdown
        )
    except Exception as e:
        st.error(f"Gagal import modul Wisdom/Insight: {e}")
        st.stop()

    df = st.session_state.df_work
    preview_df(df, "Dataset Aktif (df_work)")

    if df is None or len(df) == 0:
        st.info("Belum ada data. Upload/siapkan data terlebih dahulu.")
    else:
        # Pilih label & basis (sama logikanya dengan Page 9)
        label_candidates = [c for c in df.columns if c.lower().endswith("_label") or c.lower().startswith("cluster")]
        for c in ["KMeans_Label", "DBSCAN_Label", "Hierarchical_Label"]:
            if c in df.columns and c not in label_candidates:
                label_candidates.append(c)

        if not label_candidates:
            st.warning("Belum ada kolom label cluster. Jalankan clustering terlebih dahulu.")
        else:
            colA, colB, colC = st.columns([1,1,1])
            with colA:
                label_col = st.selectbox("Kolom label cluster", label_candidates, index=0, key="wiz_label_col")
            with colB:
                basis = st.radio("Basis interpretasi", ["Z-score (direkomendasikan)", "Mean (1–5)"], index=0, key="wiz_basis")
            with colC:
                topk = st.number_input("Maks. aspek per sisi (pos/neg)", 1, 5, 3, 1, key="wiz_topk")

            base_aspects = ["Education","Financial","Physical","Psychological","Relational"]
            use_z = basis.startswith("Z-score")
            if use_z and all((a + "_Z") in df.columns for a in base_aspects):
                aspect_cols = [a + "_Z" for a in base_aspects]
                pretty = {"Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                          "Psychological_Z":"Psikologis","Relational_Z":"Relasional"}
                lo_default, hi_default = -0.5, 0.5
            else:
                aspect_cols = [a for a in base_aspects if a in df.columns]
                pretty = {"Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                          "Psychological":"Psikologis","Relasional":"Relasional"}
                lo_default, hi_default = 2.5, 3.5

            if len(aspect_cols) < 3:
                st.warning(f"Kolom aspek kurang. Ditemukan: {aspect_cols}.")
            else:
                c1, c2 = st.columns([1,1])
                with c1:
                    low_thr = st.number_input("Ambang rendah", value=float(lo_default), step=0.1, key="wiz_thr_low")
                with c2:
                    high_thr = st.number_input("Ambang tinggi", value=float(hi_default), step=0.1, key="wiz_thr_high")

                # 1) Build profil & rekomendasi (pakai fungsi Insight)
                prof, mu, sd = cluster_profiles(df, label_col, aspect_cols, use_zscore=use_z)
                order = prof.index.astype(str).tolist()
                counts = cluster_counts(df, label_col, order=order)
                basis_name = "Z-score" if use_z else "Mean (1–5)"

                recs = make_policy_recommendations(
                    prof, counts, basis_name=basis_name,
                    high_thr=high_thr, low_thr=low_thr, top_k=int(topk)
                )

                # 2) Filter cluster yang ingin diikutkan (opsional)
                all_clusters = [r["cluster"] for r in recs]
                sel_clusters = st.multiselect("Pilih cluster yang diikutkan", all_clusters, default=all_clusters, key="wiz_sel_clusters")
                recs = [r for r in recs if r["cluster"] in set(sel_clusters)]

                # 3) Owner & timeline override (opsional)
                st.markdown("**Pengaturan umum (opsional):**")
                col1, col2 = st.columns([1,1])
                with col1:
                    owners_note = st.text_input("Owner tambahan global (opsional, pisahkan koma)", "", key="wiz_owner_add")
                with col2:
                    start = st.text_input("Mulai (mis. 'Bulan 1' atau '2025-Q1')", "Bulan 1", key="wiz_start")
                    due = st.text_input("Jatuh tempo (mis. 'Bulan 3' atau '2025-Q2')", "Bulan 3", key="wiz_due")

                override_owners = None
                if owners_note.strip():
                    # terapkan ke semua aspek sebagai tambahan
                    add = [s.strip() for s in owners_note.split(",") if s.strip()]
                    override_owners = {a.replace("_Z",""): add for a in base_aspects}

                override_timeline = (start, due) if (start and due) else None

                # 4) Bangun rencana aksi
                plan = build_action_plan(
                    recs, basis_name=basis_name, style="formal",
                    override_owners=override_owners, override_timeline=override_timeline
                )

                st.markdown("### 📋 Rencana Aksi")
                if plan is None or plan.empty:
                    st.info("Tidak ada aksi yang melewati ambang. Coba turunkan ambang atau perbesar top_k.")
                else:
                    st.dataframe(plan)

                    # Unduh CSV & Markdown
                    csv_bytes = plan.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Unduh CSV Rencana Aksi", data=csv_bytes, file_name="action_plan.csv", mime="text/csv")

                    pretty_map = {
                        "Education_Z":"Akademik","Financial_Z":"Finansial","Physical_Z":"Fisik",
                        "Psychological_Z":"Psikologis","Relational_Z":"Relasional",
                        "Education":"Akademik","Financial":"Finansial","Physical":"Fisik",
                        "Psychological":"Psikologis","Relasional":"Relasional",
                    }
                    md_text = plan_to_markdown(plan, pretty_names=pretty_map)
                    st.download_button("📝 Unduh Markdown Rencana Aksi", data=md_text.encode("utf-8"),
                                       file_name="action_plan.md", mime="text/markdown")

                st.caption("Catatan: prioritas = deviasi * sqrt(n). Pita biaya bersifat indikatif, silakan sesuaikan.")


