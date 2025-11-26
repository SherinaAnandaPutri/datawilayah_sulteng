import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="Peta Kota Kendari", layout="wide")

st.title("📍 Peta Kota Kendari + Upload Data Excel")

# ============================
# LOAD GEOJSON (LOCAL FILE)
# ============================
@st.cache_data
def load_geojson():
    with open("geojson_kendari.json", "r", encoding="utf-8") as f:
        return json.load(f)

geojson = load_geojson()

# ============================
# UPLOAD DATA EXCEL
# ============================
st.header("📤 Upload Data Excel")

uploaded = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)

    required_columns = ["kecamatan", "nilai"]

    # Validasi kolom wajib
    if not all(col in df.columns for col in required_columns):
        st.error("❌ File Excel harus memiliki kolom: **kecamatan** dan **nilai**")
    else:
        st.success("✅ File berhasil dibaca!")

        st.write("### Data Anda")
        st.dataframe(df)

        # ============================
        # PETA CHOROPLETH
        # ============================
        st.header("🗺️ Peta Kota Kendari Berdasarkan Nilai")

        fig = px.choropleth_mapbox(
            df,
            geojson=geojson,
            locations="kecamatan",
            featureidkey="properties.kecamatan",
            color="nilai",
            mapbox_style="carto-positron",
            zoom=11,
            center={"lat": -3.96, "lon": 122.52},
            color_continuous_scale="Viridis",
            opacity=0.65,
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Silakan upload file Excel berisi kolom: **kecamatan**, **nilai**")


