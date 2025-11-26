# app.py — Dashboard Kota Kendari (6 kecamatan) — stable & offline
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
import base64

st.set_page_config(page_title="Dashboard Iklim Kota Kendari", layout="wide")
st.markdown("""
<style>
body {background: linear-gradient(180deg,#f8fbff 0%, #eef6ff 100%);}
h1 {color:#0b5ed7;}
.sidebar .sidebar-content {background:#f0f6ff;}
.card {background:white; padding:12px; border-radius:8px; box-shadow: 0 6px 18px rgba(11,94,215,0.06);}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Dashboard Iklim — Kota Kendari")
st.write("Analisis tren, peta per-kecamatan, dan prediksi (2021–2070). Data default muncul otomatis; upload data opsional.")

# -------------------------
# Define kecamatan list
# -------------------------
kecamatans = ["Mandonga", "Baruga", "Kadia", "Wua-Wua", "Poasia", "Kambu"]

# -------------------------
# Embedded simplified GeoJSON for Kendari (6 kecamatan)
# NOTE: simplified polygons for visualization only (lightweight)
# -------------------------
kendari_geojson = {
    "type": "FeatureCollection",
    "features": [
        {"type":"Feature","properties":{"kecamatan":"Mandonga"},"geometry":{"type":"Polygon","coordinates":[[[122.494,-3.969],[122.500,-3.969],[122.500,-3.962],[122.494,-3.962],[122.494,-3.969]]]}},
        {"type":"Feature","properties":{"kecamatan":"Baruga"},"geometry":{"type":"Polygon","coordinates":[[[122.484,-3.976],[122.492,-3.976],[122.492,-3.968],[122.484,-3.968],[122.484,-3.976]]]}},
        {"type":"Feature","properties":{"kecamatan":"Kadia"},"geometry":{"type":"Polygon","coordinates":[[[122.499,-3.976],[122.508,-3.976],[122.508,-3.968],[122.499,-3.968],[122.499,-3.976]]]}},
        {"type":"Feature","properties":{"kecamatan":"Wua-Wua"},"geometry":{"type":"Polygon","coordinates":[[[122.497,-3.985],[122.510,-3.985],[122.510,-3.975],[122.497,-3.975],[122.497,-3.985]]]}},
        {"type":"Feature","properties":{"kecamatan":"Poasia"},"geometry":{"type":"Polygon","coordinates":[[[122.520,-3.980],[122.532,-3.980],[122.532,-3.968],[122.520,-3.968],[122.520,-3.980]]]}},
        {"type":"Feature","properties":{"kecamatan":"Kambu"},"geometry":{"type":"Polygon","coordinates":[[[122.510,-3.992],[122.524,-3.992],[122.524,-3.976],[122.510,-3.976],[122.510,-3.992]]]}}
    ]
}

# -------------------------
# Create default sample timeseries per kecamatan (2000-2020)
# -------------------------
def make_sample(seed):
    np.random.seed(seed)
    years = np.arange(2000, 2021)
    return pd.DataFrame({
        "tahun": years,
        "suhu": np.round(np.random.uniform(24.0, 29.0, len(years)) + np.linspace(0, 0.6, len(years)), 2),
        "curah_hujan": np.round(np.random.uniform(1000, 2400, len(years)) + np.linspace(0, 120, len(years)), 1),
        "kelembapan": np.round(np.random.uniform(65, 92, len(years)) + np.linspace(0, 1.2, len(years)), 1)
    })

data_by_kec = {k: make_sample(i+10) for i,k in enumerate(kecamatans)}

# -------------------------
# Sidebar controls: select kecamatan to preview & to upload data
# -------------------------
st.sidebar.header("Kontrol")
preview_kec = st.sidebar.selectbox("Pilih kecamatan untuk preview", kecamatans)
upload_kec = st.sidebar.selectbox("Jika upload data, pilih kecamatan yang akan diganti (opsional)", kecamatans)
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX untuk kecamatan terpilih (kolom: tahun,suhu,curah_hujan,kelembapan)", type=["csv","xlsx"], key="upl")

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv") or str(uploaded).lower().endswith(".csv"):
            dftmp = pd.read_csv(uploaded)
        else:
            dftmp = pd.read_excel(uploaded)
        # normalize column names
        dftmp.columns = [c.strip().lower() for c in dftmp.columns]
        mapping = {}
        # find expected columns (case insensitive)
        if "tahun" in dftmp.columns:
            mapping["tahun"] = "tahun"
        elif "year" in dftmp.columns:
            mapping["year"] = "tahun"
        # for numeric vars, try common names
        colmap = {}
        for col in dftmp.columns:
            if "suhu" in col:
                colmap[col] = "suhu"
            if "hujan" in col:
                colmap[col] = "curah_hujan"
            if "hampa" in col or "lembap" in col or "kelembap" in col:
                colmap[col] = "kelembapan"
        # build final df if contains minimal columns
        needed = ["tahun","suhu","curah_hujan","kelembapan"]
        # attempt to rename known columns
        dftmp_renamed = dftmp.rename(columns={**colmap})
        if not all(x in dftmp_renamed.columns for x in needed):
            st.sidebar.error("File harus mengandung kolom: tahun, suhu, curah_hujan, kelembapan (nama boleh variasi).")
        else:
            dftmp2 = dftmp_renamed[needed].copy()
            # ensure numeric types
            dftmp2 = dftmp2.dropna()
            dftmp2["tahun"] = pd.to_numeric(dftmp2["tahun"], errors="coerce")
            dftmp2 = dftmp2.dropna(subset=["tahun"])
            dftmp2 = dftmp2.sort_values("tahun").reset_index(drop=True)
            data_by_kec[upload_kec] = dftmp2
            st.sidebar.success(f"Data untuk {upload_kec} berhasil diperbarui ({len(dftmp2)} baris).")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file upload: {e}")

# -------------------------
# Tabs (Overview, Map, Predictions, Download)
# -------------------------
tab_overview, tab_map, tab_pred, tab_dl = st.tabs(["Overview", "Map", "Predictions", "Download"])

# -------------------------
# TAB: Overview
# -------------------------
with tab_overview:
    st.header(f"Overview — {preview_kec}")
    df_preview = data_by_kec.get(preview_kec).copy()
    st.subheader("Data (tahun, suhu, curah_hujan, kelembapan)")
    st.dataframe(df_preview, use_container_width=True)

    st.subheader("Grafik Tren")
    c1,c2 = st.columns(2)
    with c1:
        try:
            fig1 = px.line(df_preview, x="tahun", y=["suhu","kelembapan"],
                           labels={"value":"Nilai","variable":"Variabel"},
                           title=f"Suhu & Kelembapan — {preview_kec}")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuat grafik Suhu/Kelembap: {e}")
    with c2:
        try:
            fig2 = px.bar(df_preview, x="tahun", y="curah_hujan",
                          labels={"curah_hujan":"Curah Hujan (mm)"},
                          title=f"Curah Hujan Tahunan — {preview_kec}")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuat grafik Hujan: {e}")

# -------------------------
# TAB: Map
# -------------------------
with tab_map:
    st.header("Peta Kota Kendari — Perbandingan Kecamatan")
    metric_map = st.selectbox("Pilih metrik (nilai tahun terakhir tiap kecamatan)", ["suhu","curah_hujan","kelembapan"])
    rows = []
    for k in kecamatans:
        dfk = data_by_kec.get(k)
        if dfk is None or dfk.empty:
            rows.append({"kecamatan": k, "value": np.nan})
            continue
        last = dfk.iloc[-1]
        try:
            rows.append({"kecamatan": k, "value": float(last[metric_map])})
        except Exception:
            rows.append({"kecamatan": k, "value": np.nan})
    df_map = pd.DataFrame(rows)

    try:
        fig_map = px.choropleth_mapbox(
            df_map,
            geojson=kendari_geojson,
            locations="kecamatan",
            featureidkey="properties.kecamatan",
            color="value",
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            center={"lat": -3.975, "lon": 122.510},
            zoom=12,
            opacity=0.7,
            labels={"value": metric_map}
        )
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal menampilkan peta: {e}")

    st.markdown("**Tabel nilai (tahun terakhir)**")
    st.dataframe(df_map, use_container_width=True)

# -------------------------
# TAB: Predictions
# -------------------------
df_all_preds = None
with tab_pred:
    st.header("Prediksi 2021–2070 per Kecamatan (RandomForest)")

    pred_var = st.selectbox("Pilih variabel target untuk diprediksi", ["suhu","curah_hujan","kelembapan"])
    n_estimators = st.slider("Jumlah pohon RandomForest", 50, 400, 200, step=50)
    test_size = st.slider("Proporsi test untuk evaluasi", 0.05, 0.4, 0.2, step=0.05)

    all_preds = []
    metrics = []

    for k in kecamatans:
        try:
            dfi = data_by_kec.get(k)
            if dfi is None or len(dfi) < 6:
                st.warning(f"{k}: data kurang (<6). Lewati prediksi untuk kecamatan ini.")
                continue

            X = dfi[["tahun"]].values
            y = dfi[pred_var].values

            Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=test_size, random_state=42)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(Xtr, ytr)

            ypred_test = model.predict(Xts)
            rmse = np.sqrt(mean_squared_error(yts, ypred_test))
            mae = mean_absolute_error(yts, ypred_test)
            r2 = r2_score(yts, ypred_test) if len(np.unique(yts))>1 else float("nan")
            metrics.append({"kecamatan":k, "rmse":rmse, "mae":mae, "r2":r2})

            years = np.arange(2021, 2071)
            ypred = model.predict(years.reshape(-1,1))
            dfp = pd.DataFrame({"kecamatan":k, "tahun":years, f"pred_{pred_var}": ypred})
            all_preds.append(dfp)
        except Exception as e:
            st.error(f"Gagal prediksi {k}: {e}")
            continue

    if len(all_preds) == 0:
        st.info("Tidak ada prediksi karena data tidak mencukupi.")
    else:
        df_all_preds = pd.concat(all_preds, ignore_index=True)
        st.subheader("Metrik Evaluasi per Kecamatan (test split)")
        st.dataframe(pd.DataFrame(metrics).set_index("kecamatan"))

        st.subheader("Detail Prediksi — pilih kecamatan")
        pick = st.selectbox("Pilih kecamatan untuk lihat prediksi", kecamatans)
        st.write(df_all_preds[df_all_preds["kecamatan"]==pick].reset_index(drop=True))

        st.subheader("Grafik Prediksi (semua kecamatan)")
        try:
            figp = px.line(df_all_preds, x="tahun", y=f"pred_{pred_var}", color="kecamatan",
                           title=f"Prediksi {pred_var} 2021–2070 per Kecamatan")
            st.plotly_chart(figp, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal menampilkan grafik prediksi: {e}")

# -------------------------
# TAB: Download
# -------------------------
with tab_dl:
    st.header("Download Hasil Prediksi")
    if df_all_preds is None:
        st.info("Silakan buat prediksi dulu di tab 'Predictions'.")
    else:
        csv = df_all_preds.to_csv(index=False)
        st.download_button("Download CSV prediksi semua kecamatan", csv, file_name="prediksi_kendari_2021_2070.csv", mime="text/csv")

        # Excel multi-sheet
        try:
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                for k in kecamatans:
                    dfx = df_all_preds[df_all_preds["kecamatan"]==k].reset_index(drop=True)
                    sheet = k[:31]  # sheet name limit
                    dfx.to_excel(writer, sheet_name=sheet, index=False)
            towrite.seek(0)
            b64 = base64.b64encode(towrite.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediksi_kendari_2021_2070.xlsx">Download Excel (per-kecamatan)</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal membuat file Excel: {e}")

st.markdown("---")
st.caption("Dashboard Kota Kendari — versi stabil. Polygon peta disederhanakan untuk performa; untuk peta akurat, minta GeoJSON resmi.")
