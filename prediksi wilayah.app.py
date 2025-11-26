import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="Peta Sulawesi Tenggara – Kendari", layout="wide")

st.title("🗺️ Dashboard Peta Kota Kendari – Sulawesi Tenggara")

st.header("📤 Upload Data Excel")
file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

# ======================
# Load GeoJSON Kendari
# ======================
@st.cache_data
def load_geojson():
    with open("geojson_kendari.json", "r", encoding="utf-8") as f:
        return json.load(f)

geojson = load_geojson()


# ======================
# Jika file Excel di-upload
# ======================
if file:
    df = pd.read_excel(file)

    st.subheader("📄 Data yang Diunggah")
    st.dataframe(df)

    # normalisasi kolom
    kolom = [c.lower().strip() for c in df.columns]

    def cari_kolom(target, kolom):
        for c in kolom:
            if target in c:
                return c
        return None

    kol_kecamatan = cari_kolom("kecamatan", kolom)
    kol_nilai = cari_kolom("nilai", kolom)

    if kol_kecamatan is None or kol_nilai is None:
        st.error("❌ File Excel harus memiliki kolom yang berisi teks 'kecamatan' dan 'nilai'.")
        st.stop()

    df.columns = kolom
    df = df[[kol_kecamatan, kol_nilai]]
    df.columns = ["kecamatan", "nilai"]

    st.success("✅ Kolom berhasil dibaca!")

    st.subheader("🌍 Peta Kota Kendari (Choropleth)")
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="kecamatan",
        featureidkey="properties.Kecamatan",
        color="nilai",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": -3.995, "lon": 122.518},
        opacity=0.7,
        labels={'nilai': 'Nilai'},
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Silakan unggah file Excel untuk melanjutkan.")
