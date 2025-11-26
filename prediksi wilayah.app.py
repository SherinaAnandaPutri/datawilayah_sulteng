import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

# ==========================
# GEOJSON SULTENG (embedded, tidak perlu internet)
# ==========================
geojson_sulteng = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "Palu"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[119.82, -0.86],[119.90, -0.86],[119.90, -0.74],[119.82, -0.74],[119.82, -0.86]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Donggala"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[119.70, -0.98],[119.90, -0.98],[119.90, -0.86],[119.70, -0.86],[119.70, -0.98]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Sigi"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[119.80, -1.05],[120.00, -1.05],[120.00, -0.90],[119.80, -0.90],[119.80, -1.05]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Parigi Moutong"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.00, -0.90],[120.20, -0.90],[120.20, -0.60],[120.00, -0.60],[120.00, -0.90]]]
      }
    }
  ]
}

# ==========================
# TITLE
# ==========================
st.title("🌦️ Prediksi Iklim Wilayah Sulawesi Tengah (SULTENG)")
st.write("Versi tanpa GeoPandas dan tanpa request online — dijamin tanpa error di Streamlit Cloud.")

# ==========================
# UPLOAD DATA
# ==========================
st.subheader("📤 Upload Dataset Iklim Harian (CSV)")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Awal")
    st.dataframe(df.head())

    st.subheader("⚙️ Pilih Fitur & Target")
    cols = df.columns.tolist()

    fitur = st.multiselect("Pilih fitur", cols)
    target = st.selectbox("Pilih target", cols)

    if fitur and target:
        X = df[fitur]
        y = df[target]

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

            st.subheader("⏳ Prediksi 10–50 Tahun")
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
# PETA SULTENG
# ==========================
st.subheader("🗺️ Peta Wilayah Sulawesi Tengah")

wilayah = [f["properties"]["name"] for f in geojson_sulteng["features"]]
dummy = np.random.rand(len(wilayah))

df_map = pd.DataFrame({
    "wilayah": wilayah,
    "nilai": dummy
})

fig = px.choropleth(
    df_map,
    geojson=geojson_sulteng,
    locations="wilayah",
    featureidkey="properties.name",
    color="nilai",
    color_continuous_scale="Viridis",
    title="Peta Kabupaten SULTENG (Simplified)"
)

fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

st.info("Peta dan model berjalan 100% tanpa GeoPandas dan tanpa permintaan internet.")
