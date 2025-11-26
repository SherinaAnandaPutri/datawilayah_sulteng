# app.py (perbaikan: tidak memakai mean_squared_error(..., squared=False))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
import base64

st.set_page_config(page_title="Dashboard Multi-Wilayah SULTENG (7 wilayah)", layout="wide")
st.markdown("""
<style>
body {background: linear-gradient(180deg, #f6fbff 0%, #eef6ff 100%);}
h1 {color: #0b5ed7;}
.sidebar .sidebar-content {background: #e9f2ff;}
.metric {background: white; padding: 12px; border-radius:10px; box-shadow: 0 4px 14px rgba(11,94,215,0.06);}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Dashboard Multi-Wilayah — SULTENG (7 Wilayah)")
st.write("Default data otomatis tersedia untuk 7 wilayah. Kamu bisa mengganti data per-wilayah dengan upload Excel jika ingin.")

# --------------------------
# 1) GeoJSON embedded (7 wilayah simplified)
# --------------------------
geojson_multi = {
    "type": "FeatureCollection",
    "features": [
        {"type":"Feature","properties":{"name":"SULTENG(1)"},"geometry":{"type":"Polygon","coordinates":[[[120.0,-2.0],[121.0,-2.0],[121.0,-1.2],[120.0,-1.2],[120.0,-2.0]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(2)"},"geometry":{"type":"Polygon","coordinates":[[[121.0,-2.2],[122.0,-2.2],[122.0,-1.4],[121.0,-1.4],[121.0,-2.2]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(3)"},"geometry":{"type":"Polygon","coordinates":[[[119.0,-1.8],[120.0,-1.8],[120.0,-1.0],[119.0,-1.0],[119.0,-1.8]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(4)"},"geometry":{"type":"Polygon","coordinates":[[[121.5,-1.2],[122.5,-1.2],[122.5,-0.4],[121.5,-0.4],[121.5,-1.2]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(5)"},"geometry":{"type":"Polygon","coordinates":[[[120.5,-0.8],[121.5,-0.8],[121.5,-0.0],[120.5,-0.0],[120.5,-0.8]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(6)"},"geometry":{"type":"Polygon","coordinates":[[[122.0,-0.6],[123.0,-0.6],[123.0,0.2],[122.0,0.2],[122.0,-0.6]]]}},
        {"type":"Feature","properties":{"name":"SULTENG(7)"},"geometry":{"type":"Polygon","coordinates":[[[121.0,0.0],[122.0,0.0],[122.0,0.8],[121.0,0.8],[121.0,0.0]]]}}
    ]
}

# --------------------------
# 2) Default sample datasets for 7 regions (2000-2020)
# --------------------------
regions = [f"SULTENG({i})" for i in range(1,8)]

def make_sample_df(seed):
    np.random.seed(seed)
    years = np.arange(2000, 2021)
    return pd.DataFrame({
        "tahun": years,
        "suhu": np.round(np.random.uniform(24.0, 29.0, len(years)) + np.linspace(0, 0.8, len(years)), 2),
        "hujan": np.round(np.random.uniform(1000, 2400, len(years)) + np.linspace(0, 150, len(years)), 1),
        "lembap": np.round(np.random.uniform(65, 92, len(years)) + np.linspace(0, 1.5, len(years)), 1)
    })

default_data = {r: make_sample_df(i) for i,r in enumerate(regions, start=1)}

# --------------------------
# Sidebar: choose region to view / upload
# --------------------------
st.sidebar.header("🔧 Kontrol Dashboard")
selected_region = st.sidebar.selectbox("Pilih wilayah (preview & prediksi)", regions)
upload_region = st.sidebar.selectbox("Jika upload data, pilih wilayah yang akan diganti (opsional)", regions)
uploaded_file = st.sidebar.file_uploader("Upload Excel untuk wilayah terpilih (kolom: tahun,suhu,hujan,lembap)", type=["xlsx","csv"], key="uploader")

if uploaded_file:
    try:
        if str(uploaded_file).lower().endswith(".csv") or uploaded_file.name.lower().endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        # normalize columns
        cols_lower = {c: c.strip().lower() for c in df_upload.columns}
        df_upload.rename(columns=cols_lower, inplace=True)
        # ensure needed cols exist
        needed = ["tahun","suhu","hujan","lembap"]
        if not all(c in df_upload.columns for c in needed):
            st.sidebar.error("File harus memiliki kolom: tahun, suhu, hujan, lembap (nama bisa beda kapitalisasi).")
            df_upload = None
        else:
            df_upload = df_upload[["tahun","suhu","hujan","lembap"]]
            df_upload = df_upload.sort_values("tahun").reset_index(drop=True)
            default_data[upload_region] = df_upload
            st.sidebar.success(f"Data untuk {upload_region} berhasil diperbarui dari upload.")
    except Exception as e:
        st.sidebar.error("Gagal membaca file upload — pastikan format Excel/CSV benar.")

# --------------------------
# Tabs: Overview / Map / Predictions / Download
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🗺️ Map", "🔮 Predictions", "📥 Download"])

# --------------------------
# TAB 1: Overview - show selected region data and trends
# --------------------------
with tab1:
    st.header(f"Overview — {selected_region}")
    df_region = default_data[selected_region].copy()
    st.subheader("Data (tahun, suhu, hujan, lembap)")
    st.dataframe(df_region, use_container_width=True)

    st.subheader("Grafik Tren")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(df_region, x="tahun", y=["suhu","lembap"], markers=True,
                       labels={"value":"Nilai","variable":"Variabel"}, title=f"Suhu & Kelembapan — {selected_region}")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.bar(df_region, x="tahun", y="hujan", labels={"hujan":"Curah Hujan (mm)"}, title=f"Curah Hujan Tahunan — {selected_region}")
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# TAB 2: Map - show choropleth of latest-year metric across regions
# --------------------------
with tab2:
    st.header("Peta — Perbandingan Wilayah")
    metric = st.selectbox("Pilih metrik untuk peta (menggunakan nilai tahun terakhir tiap wilayah)", ["suhu","hujan","lembap"], index=0)

    # prepare map dataframe: for each region take last year's value
    map_rows = []
    for r in regions:
        rd = default_data[r]
        last = rd.iloc[-1]
        map_rows.append({"region": r, "value": float(last[metric])})
    df_map = pd.DataFrame(map_rows)

    fig_map = px.choropleth_mapbox(
        df_map,
        geojson=geojson_multi,
        locations="region",
        featureidkey="properties.name",
        color="value",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        center={"lat": -1.0, "lon": 121.0},
        zoom=6,
        opacity=0.7,
        labels={"value": metric}
    )
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("**Tabel nilai (tahun terakhir)**")
    st.dataframe(df_map, use_container_width=True)

# --------------------------
# TAB 3: Predictions - build per-region RF models and show preds 2021-2070
# --------------------------
with tab3:
    st.header("Prediksi 2021–2070 per Wilayah (RandomForest)")

    pred_target = st.selectbox("Pilih target prediksi", ["suhu","hujan","lembap"], index=0)
    n_estimators = st.slider("Jumlah pohon RandomForest", 50, 500, 200, step=50)
    test_size = st.slider("Proporsi test (untuk evaluasi)", 0.05, 0.4, 0.2, step=0.05)

    all_preds = []
    metrics_summary = []

    for r in regions:
        rd = default_data[r].copy()
        X = rd[["tahun"]].values
        y = rd[pred_target].values
        if len(rd) < 6:
            st.warning(f"{r}: data terlalu sedikit ({len(rd)} baris), melewatkan prediksi.")
            continue
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=test_size, random_state=42)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(Xtr, ytr)
        ypred_test = model.predict(Xts)

        # Perbaikan: hitung RMSE manual (compatibility across sklearn versions)
        rmse = np.sqrt(mean_squared_error(yts, ypred_test))
        mae = mean_absolute_error(yts, ypred_test)
        r2 = r2_score(yts, ypred_test) if len(np.unique(yts))>1 else float("nan")
        metrics_summary.append({"region":r, "rmse":rmse, "mae":mae, "r2":r2})

        years = np.arange(2021, 2071)
        ypred = model.predict(years.reshape(-1,1))
        dfp = pd.DataFrame({"region": r, "tahun": years, f"pred_{pred_target}": ypred})
        all_preds.append(dfp)

    if len(all_preds)==0:
        st.error("Tidak ada prediksi yang berhasil dibuat (cek data wilayah).")
    else:
        df_all_preds = pd.concat(all_preds, ignore_index=True)
        st.subheader("Ringkasan metrik per wilayah (evaluasi test split)")
        st.dataframe(pd.DataFrame(metrics_summary).set_index("region"))

        st.subheader("Contoh Prediksi (pilih wilayah untuk melihat detail)")
        chosen = st.selectbox("Pilih wilayah untuk detail prediksi", regions, index=0)
        st.write(df_all_preds[df_all_preds["region"]==chosen].reset_index(drop=True))

        st.subheader("Grafik Prediksi (semua wilayah)")
        fig_preds = px.line(df_all_preds, x="tahun", y=f"pred_{pred_target}", color="region",
                            title=f"Prediksi {pred_target} 2021–2070 (per wilayah)")
        st.plotly_chart(fig_preds, use_container_width=True)

# --------------------------
# TAB 4: Download - export combined predictions
# --------------------------
with tab4:
    st.header("Download Hasil Prediksi")

    if 'df_all_preds' not in locals():
        st.info("Silakan buat prediksi dulu di tab 'Predictions'.")
    else:
        st.write("Unduh gabungan seluruh prediksi (CSV / Excel).")
        csv = df_all_preds.to_csv(index=False)
        st.download_button("🔽 Download CSV prediksi semua wilayah", csv, file_name="prediksi_sulteng_multi_2021_2070.csv", mime="text/csv")

        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            for r in regions:
                dfp = df_all_preds[df_all_preds["region"]==r].reset_index(drop=True)
                dfp.to_excel(writer, sheet_name=r[:31], index=False)
            writer.save()
            towrite.seek(0)
            b64 = base64.b64encode(towrite.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediksi_sulteng_multi_2021_2070.xlsx">🔽 Download Excel prediksi semua wilayah</a>'
            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("Dashboard multi-wilayah SULTENG (7 wilayah) — sample data dibuat acak untuk demo. "
           "Kamu bisa upload data per-wilayah di sidebar untuk mengganti data default.")
