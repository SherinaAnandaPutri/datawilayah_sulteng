import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ----------------------------------------------------------
st.set_page_config(page_title="Dashboard Iklim Kendari", layout="wide")

st.title("🌦️ Dashboard Analisis & Prediksi Iklim — Kota Kendari")
st.write("Wilayah: Mandonga • Baruga • Kadia • Wua-Wua • Poasia • Kambu")

# ----------------------------------------------------------
# 2. DATA KECAMATAN (Marker Map)
# ----------------------------------------------------------
kecamatan_locs = {
    "Mandonga": [-3.967, 122.514],
    "Baruga": [-4.005, 122.495],
    "Kadia": [-3.996, 122.529],
    "Wua-Wua": [-4.003, 122.523],
    "Poasia": [-4.030, 122.542],
    "Kambu": [-4.020, 122.509],
}

# ----------------------------------------------------------
# 3. GENERATE DATA IKLIM OTOMATIS (TANPA UPLOAD)
# ----------------------------------------------------------
def generate_weather_data(seed=0):
    np.random.seed(seed)
    years = range(2010, 2025)
    months = range(1, 13)

    records = []

    for y in years:
        for m in months:
            records.append([
                y, m,
                np.random.uniform(22, 25),   # Tn
                np.random.uniform(29, 33),   # Tx
                np.random.uniform(25, 28),   # Tavg
                np.random.uniform(60, 90),   # Kelembaban
                np.random.uniform(50, 300),  # Curah hujan
                np.random.uniform(4, 10),    # Matahari
                np.random.uniform(1, 5),     # Angin
            ])
    
    df = pd.DataFrame(records, columns=[
        "Tahun","Bulan","Tn","Tx","Tavg","Kelembaban",
        "Curah_Hujan","Matahari","Angin"
    ])
    return df

df = generate_weather_data()

# ----------------------------------------------------------
# 4. PETA KENDARI
# ----------------------------------------------------------
st.subheader("🗺️ Peta Persebaran Kecamatan di Kota Kendari")

map_center = [-4.00, 122.52]
m = folium.Map(location=map_center, zoom_start=12)

for kec, (lat, lon) in kecamatan_locs.items():
    folium.Marker(
        location=[lat, lon],
        popup=kec,
        tooltip=f"Kecamatan {kec}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

st_folium(m, width=800, height=420)

# ----------------------------------------------------------
# 5. GRAFIK TREN HISTORIS
# ----------------------------------------------------------
st.subheader("📈 Tren Data Historis (2010–2024)")

df["Tanggal"] = pd.to_datetime(df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-01")

var_options = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "Kelembaban": "Kelembaban Udara (%)",
    "Curah_Hujan": "Curah Hujan (mm)",
    "Matahari": "Durasi Penyinaran (jam)",
    "Angin": "Kecepatan Angin (m/s)"
}

selected_var = st.selectbox("Pilih Variabel", list(var_options.keys()), format_func=lambda x: var_options[x])

fig_hist = px.line(
    df,
    x="Tanggal",
    y=selected_var,
    title=f"Tren {var_options[selected_var]}",
    markers=True
)
st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------------
# 6. MODEL & PREDIKSI 2025–2075
# ----------------------------------------------------------
st.subheader("🔮 Prediksi Iklim Kota Kendari (2025–2075)")

models = {}

for col in ["Tn","Tx","Tavg","Kelembaban","Curah_Hujan","Matahari","Angin"]:
    X = df[["Tahun","Bulan"]]
    y = df[col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    models[col] = model

# future data
future = pd.DataFrame([(y,m) for y in range(2025,2076) for m in range(1,13)], columns=["Tahun","Bulan"])
future["Tanggal"] = pd.to_datetime(future["Tahun"].astype(str)+"-"+future["Bulan"].astype(str)+"-01")

for col in models:
    future[f"Pred_{col}"] = models[col].predict(future[["Tahun","Bulan"]])

# pilih variabel prediksi
selected_pred = st.selectbox("Pilih Variabel Prediksi", list(var_options.keys()), format_func=lambda x: var_options[x])

fig_pred = px.line(
    future,
    x="Tanggal",
    y=f"Pred_{selected_pred}",
    title=f"Prediksi {var_options[selected_pred]} (2025–2075)"
)
st.plotly_chart(fig_pred, use_container_width=True)

# ----------------------------------------------------------
# 7. DOWNLOAD DATA
# ----------------------------------------------------------
st.subheader("📥 Download Dataset Prediksi")
csv = future.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Prediksi Kendari",
    data=csv,
    file_name="prediksi_kendari_2025_2075.csv",
    mime="text/csv"
)

st.success("Dashboard berhasil dimuat tanpa error!")
