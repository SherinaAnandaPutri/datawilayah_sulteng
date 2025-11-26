import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="Dashboard Iklim Kendari", layout="wide")

st.title("📊 Dashboard Analisis & Peta Kota Kendari — Sulawesi Tenggara")

# -----------------------------------------------------------
# 1. GEOJSON KENDARI (SIMPIFIED – TANPA DOWNLOAD)
# -----------------------------------------------------------
kendari_geojson = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"kecamatan": "Mandonga"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.496, -3.968],[122.508, -3.968],[122.508, -3.957],[122.496, -3.957],[122.496, -3.968]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"kecamatan": "Baruga"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.485, -3.975],[122.496, -3.975],[122.496, -3.963],[122.485, -3.963],[122.485, -3.975]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"kecamatan": "Kadia"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.500, -3.975],[122.510, -3.975],[122.510, -3.965],[122.500, -3.965],[122.500, -3.975]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"kecamatan": "Wua-Wua"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.496, -3.985],[122.510, -3.985],[122.510, -3.975],[122.496, -3.975],[122.496, -3.985]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"kecamatan": "Poasia"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.520, -3.980],[122.535, -3.980],[122.535, -3.965],[122.520, -3.965],[122.520, -3.980]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"kecamatan": "Kambu"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[122.510, -3.990],[122.525, -3.990],[122.525, -3.975],[122.510, -3.975],[122.510, -3.990]]]
      }
    }
  ]
}

st.header("📤 Upload Data Excel")

uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # VALIDASI
    required_cols = ["kecamatan", "nilai"]
    if not all(c in df.columns for c in required_cols):
        st.error("❌ File Excel harus memiliki kolom: **kecamatan** dan **nilai**")
        st.stop()

    st.success("✔ File berhasil dibaca!")
    st.write("### 📄 Data Anda:")
    st.dataframe(df)

    # -----------------------------------------------------------
    # 2. MENAMPILKAN PETA KENDARI BERDASARKAN DATA
    # -----------------------------------------------------------
    st.header("🗺️ Peta Kota Kendari Berdasarkan Data Excel")

    fig = px.choropleth_mapbox(
        df,
        geojson=kendari_geojson,
        locations="kecamatan",
        featureidkey="properties.kecamatan",
        color="nilai",
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": -3.97, "lon": 122.51},
        color_continuous_scale="Viridis",
        opacity=0.7
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬆ Silakan upload file Excel untuk memulai.")
