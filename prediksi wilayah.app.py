import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd

st.set_page_config(page_title="🌦️ Dashboard Iklim SULTENG(1)", layout="wide")

# =============================
# 1. LOAD DATA UTAMA SULTENG
# =============================
@st.cache_data
def load_data():
    df = pd.read_excel("Data SULTENG(1).xlsx", sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin":"FF_X"})
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

with st.spinner("⏳ Memuat Data Iklim SULTENG..."):
    df = load_data()

wilayah = "Sulawesi Tengah"
st.title(f"🌤️ Dashboard Analisis & Prediksi Iklim — {wilayah}")
st.markdown("### Analisis cuaca historis, prediksi 50 tahun, dan peta geospasial kabupaten di Sulteng.")

# ===================================
# 2. DATA GEOSPASIAL PETA SULTENG
# ===================================
@st.cache_data
def load_geo():
    # file harus disediakan di folder kerja
    return gpd.read_file("sulteng_shapefile.geojson")

try:
    geo = load_geo()
    peta_siap = True
except:
    peta_siap = False

# =============================
# 3. SIDEBAR
# =============================
st.sidebar.header("🔎 Filter Data Historis")

selected_year = st.sidebar.multiselect(
    "Pilih Tahun", sorted(df["Tahun"].unique()), default=df["Tahun"].unique()
)
selected_month = st.sidebar.multiselect(
    "Pilih Bulan", list(range(1,13)), default=list(range(1,13))
)

df = df[df["Tahun"].isin(selected_year)]
df = df[df["Bulan"].isin(selected_month)]

# =============================
# 4. VARIABEL
# =============================
possible_vars = ["Tn","Tx","Tavg","kelembaban","curah_hujan","matahari","FF_X","DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn":"Suhu Minimum (°C)",
    "Tx":"Suhu Maksimum (°C)",
    "Tavg":"Suhu Rata-rata (°C)",
    "kelembaban":"Kelembaban (%)",
    "curah_hujan":"Curah Hujan (mm)",
    "matahari":"Durasi Penyinaran Matahari (jam)",
    "FF_X":"Kecepatan Angin (m/s)",
    "DDD_X":"Arah Angin (°)"
}

# =============================
# 5. AGREGASI HISTORIS
# =============================
agg = {v:"mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun","Bulan"]).agg(agg).reset_index()
monthly["Tanggal"] = pd.to_datetime(monthly["Tahun"].astype(str) + "-" + monthly["Bulan"].astype(str) + "-01")

# =============================
# 6. KARTU STATISTIK
# =============================
c1,c2,c3 = st.columns(3)
c1.metric("📏 Total Record", f"{len(df):,}")
c2.metric("📅 Rentang Tahun", f"{df['Tahun'].min()}–{df['Tahun'].max()}")
c3.metric("📦 Variabel Iklim", len(available_vars))

# =============================
# 7. GRAFIK HISTORIS
# =============================
st.subheader("📈 Tren Data Historis")
var_hist = st.selectbox("Pilih Variabel", [label[v] for v in available_vars])
key_hist = [k for k,v in label.items() if v==var_hist][0]

fig_hist = px.line(
    monthly,
    x="Tanggal", y=key_hist,
    title=f"Tren {var_hist} di {wilayah}",
    markers=True, template="plotly_white",
    line_shape="spline"
)
st.plotly_chart(fig_hist, use_container_width=True)

# =============================
# 8. MODEL & PREDIKSI 50 TAHUN
# =============================
st.subheader("🔮 Prediksi Iklim 2025–2075")
models = {}

for v in available_vars:
    X = monthly[["Tahun","Bulan"]]
    y = monthly[v]
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)
    models[v] = model

future = pd.DataFrame([(y,m) for y in range(2025,2076) for m in range(1,13)], columns=["Tahun","Bulan"])
for v in available_vars:
    future[f"Pred_{v}"] = models[v].predict(future[["Tahun","Bulan"]])

future["Tanggal"] = pd.to_datetime(future["Tahun"].astype(str)+"-"+future["Bulan"].astype(str)+"-01")

var_pred = st.selectbox("Pilih Variabel Prediksi", [label[v] for v in available_vars])
key_pred = [k for k,v in label.items() if v==var_pred][0]

fig_pred = px.line(
    future, x="Tanggal", y=f"Pred_{key_pred}",
    title=f"Prediksi {var_pred} 2025–2075", template="plotly_white", line_shape="spline"
)
st.plotly_chart(fig_pred, use_container_width=True)

# =============================
# 9. PETA GEOSPASIAL KABUPATEN
# =============================
st.subheader("🗺️ Peta Sebaran Iklim di Kabupaten Sulteng")

if peta_siap:
    var_map = st.selectbox("Pilih Variabel untuk Dipetakan", [label[v] for v in available_vars])
    key_map = [k for k,v in label.items() if v==var_map][0]

    # hitung rata-rata per tahun terakhir
    latest_year = df["Tahun"].max()
    df_latest = df[df["Tahun"] == latest_year]
    df_kab = df_latest.groupby("Stasiun").agg({key_map:"mean"}).reset_index()

    geo_merge = geo.merge(df_kab, left_on="nama_kabupaten", right_on="Stasiun", how="left")

    fig_map = px.choropleth_mapbox(
        geo_merge,
        geojson=geo_merge.geometry,
        locations=geo_merge.index,
        color=key_map,
        mapbox_style="carto-positron",
        zoom=5.2,
        center={"lat": -1.0, "lon": 121.6},
        opacity=0.5,
        title=f"Peta {var_map} — Tahun {latest_year}"
    )
    st.plotly_chart(fig_map, use_container_width=True)

else:
    st.warning("⚠️ File peta 'sulteng_shapefile.geojson' tidak ditemukan. Silakan upload untuk menampilkan peta.")

# =============================
# 10. DOWNLOAD DATA PREDIKSI
# =============================
csv = future.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Dataset Prediksi 2025–2075", data=csv,
    file_name="prediksi_sulteng_2025_2075.csv", mime="text/csv"
)
