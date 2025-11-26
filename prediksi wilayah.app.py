import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# 🎨 Custom Theme & Styling
# =========================================================
st.set_page_config(
    page_title="📊 Dashboard Prediksi Iklim",
    layout="wide",
    page_icon="🌦️"
)

# CSS PREMIUM
st.markdown("""
<style>
/* Background soft gradient */
.main {
    background: linear-gradient(to bottom right, #f7faff, #eef2ff);
}

/* Card Styling */
.metric-card {
    padding: 20px;
    border-radius: 14px;
    background: white;
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    text-align: center;
    transition: 0.3s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.10);
}

/* Section title */
h2 {
    color: #1f3a93;
    font-weight: 700;
    margin-top: 20px;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: #f0f4ff !important;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #6c6c6c;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Data SULTENG(1).xlsx", sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

with st.spinner("⏳ Sedang memuat data..."):
    df = load_data()

wilayah = "Jawa Timur"
st.title(f"🌦️ Dashboard Analisis & Prediksi Iklim — {wilayah}")
st.write("Visualisasi data historis dan prediksi iklim jangka panjang (2025–2075).")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🔧 Filter Data")
st.sidebar.divider()

selected_year = st.sidebar.multiselect(
    "🗓️ Pilih Tahun",
    sorted(df["Tahun"].unique()),
    default=df["Tahun"].unique()
)

selected_month = st.sidebar.multiselect(
    "📅 Pilih Bulan",
    range(1, 13),
    default=range(1, 13)
)

df = df[df["Tahun"].isin(selected_year)]
df = df[df["Bulan"].isin(selected_month)]

possible_vars = ["Tn","Tx","Tavg","kelembaban","curah_hujan","matahari","FF_X","DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Matahari (jam)",
    "FF_X": "Kecepatan Angin (m/s)",
    "DDD_X": "Arah Angin (°)"
}

# =========================================================
# AGREGASI
# =========================================================
agg_dict = {v: "mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun", "Bulan"]).agg(agg_dict).reset_index()

# =========================================================
# MODEL
# =========================================================
models = {}
metrics = {}

for v in available_vars:
    X = monthly[["Tahun","Bulan"]]
    y = monthly[v]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

