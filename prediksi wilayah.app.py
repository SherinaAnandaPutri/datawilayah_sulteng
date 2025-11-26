import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="📊 Dashboard Iklim Kendari", layout="wide")

# ======================================================================
# PENJELASAN Riwayat Nama:
#
# Pada versi awal dashboard (versi sementara), wilayah belum dinamai
# sehingga diberi placeholder:
#
#   Mandonga
#   Baruga
#   Kadia
#   Wua-Wua
#   Poasia
#   Kambu
#   Abeli (opsional)
#
# Sekarang seluruh placeholder sudah diganti dengan NAMA ASLI kecamatan
# di Kota Kendari. Tidak ada lagi penggunaan Sulteng 1–7.
# Dokumentasi ini hanya untuk menghindari kebingungan versi lama.
# ======================================================================


# ============================
# 1. Daftar Kecamatan
# ============================
kecamatans = ["Mandonga", "Baruga", "Kadia", "Wua-Wua", "Poasia", "Kambu"]


# ============================
# 2. Lokasi Koordinat Marker Map
# (bukan polygon, agar ringan dan tidak error)
# ============================
lokasi_map = {
    "Mandonga": [-3.9673, 122.5148],
    "Baruga": [-3.9909, 122.5313],
    "Kadia": [-3.9823, 122.5075],
    "Wua-Wua": [-3.9952, 122.5150],
    "Poasia": [-4.0163, 122.5590],
    "Kambu": [-3.9877, 122.5321],
}


# ============================
# 3. Membuat Data Iklim Otomatis
# (Tidak perlu upload Excel)
# ============================
def generate_data(n=365):
    np.random.seed(42)
    return pd.DataFrame({
        "Tanggal": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Tavg": np.random.uniform(25, 31, n),
        "Tmin": np.random.uniform(23, 28, n),
        "Tmax": np.random.uniform(28, 35, n),
        "Kelembaban": np.random.uniform(60, 95, n),
        "Curah_Hujan": np.random.uniform(0, 30, n),
        "Kecepatan_Angin": np.random.uniform(1, 6, n),
    })


# Buat data otomatis untuk tiap kecamatan
data_kecamatan = {k: generate_data() for k in kecamatans}


st.title("🌦️ Dashboard Iklim Kota Kendari")
st.write("Menampilkan data iklim otomatis untuk 6 kecamatan tanpa upload file.")


# ============================
# 4. Sidebar
# ============================
st.sidebar.header("Pilih Wilayah")
selected_kec = st.sidebar.selectbox("Kecamatan", kecamatans)


# ============================
# 5. Menampilkan Data
# ============================
st.subheader(f"📌 Data Iklim — {selected_kec}")
df = data_kecamatan[selected_kec]
st.dataframe(df, use_container_width=True)


# ============================
# 6. Grafik Tren
# ============================
st.subheader("📈 Grafik Tren Iklim")

var = st.selectbox("Pilih Variabel", ["Tavg", "Tmin", "Tmax", "Kelembaban", "Curah_Hujan", "Kecepatan_Angin"])

fig = px.line(df, x="Tanggal", y=var, title=f"Tren {var} — {selected_kec}", markers=True)
st.plotly_chart(fig, use_container_width=True)


# ============================
# 7. Peta Marker
# ============================
st.subheader("🗺️ Peta Kecamatan Kota Kendari")

map_df = pd.DataFrame({
    "Kecamatan": kecamatans,
    "lat": [lokasi_map[k][0] for k in kecamatans],
    "lon": [lokasi_map[k][1] for k in kecamatans],
})

fig_map = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    color="Kecamatan",
    zoom=11,
    size_max=18,
    mapbox_style="open-street-map",
)
st.plotly_chart(fig_map, use_container_width=True)


# ============================
# 8. Download Data
# ============================
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Data Kecamatan",
    csv,
    f"data_{selected_kec}.csv",
    "text/csv"
)
