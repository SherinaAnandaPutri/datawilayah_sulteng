import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests

# ==========================
# TITLE
# ==========================
st.title("🌦️ Prediksi Iklim Wilayah Sulawesi Tengah (SULTENG)")
st.write("Aplikasi prediksi iklim berbasis Machine Learning yang berfungsi tanpa GeoPandas.")

# ==========================
# LOAD GEOJSON SULTENG
# ==========================
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/samagra14/indonesia-geojson/master/kabupaten/sulawesi_tengah.geojson"
    response = requests.get(url)
    return response.json()

geojson_sulteng = load_geojson()

# ==========================
# UPLOAD DATASET
# ==========================
st.subheader("📤 Upload Dataset Iklim Harian (CSV)")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 🔍 Data Awal")
    st.dataframe(df.head())

    # ==========================
    # SELECT FEATURES & TARGET
    # ==========================
    st.subheader("⚙️ Pilih Fitur & Target Prediksi")
    cols = df.columns.tolist()

    fitur = st.multiselect("Pilih fitur", cols)
    target = st.selectbox("Pilih target", cols)

    if fitur and target:
        X = df[fitur]
        y = df[target]

        # ==========================
        # TRAIN MODEL
        # ==========================
        test_size = st.slider("Proporsi Data Test", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 9999, 42)

        if st.button("🚀 Latih Model"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            st.success("Model berhasil dilatih!")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**R2 Score:** {r2:.4f}")

            # ==========================
            # PREDIKSI 10–50 TAHUN
            # ==========================
            st.subheader("⏳ Prediksi Iklim 10–50 Tahun")
            future_years = st.slider("Berapa tahun ke depan?", 10, 50, 30)

            X_future = np.tile(X.mean().values, (future_years, 1))
            future_pred = model.predict(X_future)

            tahun = np.arange(1, future_years + 1)

            df_future = pd.DataFrame({
                "Tahun ke-": tahun,
                f"Prediksi {target}": future_pred
            })

            st.write("### 🔮 Hasil Prediksi")
            st.line_chart(df_future.set_index("Tahun ke-"))

# ==========================
# PETA SULTENG DENGAN GEOJSON
# ==========================
st.subheader("🗺️ Peta Wilayah Sulawesi Tengah")

# Ambil nama wilayah
nama_wilayah = [f["properties"]["name"] for f in geojson_sulteng["features"]]
nilai_dummy = np.random.rand(len(nama_wilayah))

df_map = pd.DataFrame({
    "wilayah": nama_wilayah,
    "nilai": nilai_dummy
})

fig = px.choropleth(
    df_map,
    geojson=geojson_sulteng,
    locations="wilayah",
    featureidkey="properties.name",
    color="nilai",
    color_continuous_scale="Viridis",
    title="Peta Kabupaten di Sulawesi Tengah"
)

fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

st.info("Peta aktif tanpa GeoPandas. Semua berjalan ringan di Streamlit Cloud.")
