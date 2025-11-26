import streamlit as st
import pandas as pd
import json
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Iklim SULTENG(1)", layout="wide")

st.title("🌦️ Prediksi Iklim SULTENG(1)")
st.write("Dashboard untuk upload data Excel, visualisasi, dan prediksi iklim menggunakan Machine Learning.")

# ===========================
# GEOJSON SULTENG(1) (EMBED)
# ===========================
geojson_sulteng = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"NAME": "SULTENG(1)", "id": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [121.0, -1.0],
                    [122.2, -1.3],
                    [123.6, -0.9],
                    [122.8, 0.2],
                    [121.5, 0.3],
                    [121.0, -1.0]
                ]]
            }
        }
    ]
}

# Upload Data
st.subheader("📤 Upload Data Excel Harian Iklim")
uploaded_file = st.file_uploader("Upload file Excel (*.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File berhasil diupload!")

    st.subheader("📊 Data Preview")
    st.dataframe(df)

    # ----------------------------------
    # VISUALISASI PETA
    # ----------------------------------
    st.subheader("🗺️ Peta Wilayah SULTENG(1)")

    fig_map = px.choropleth_mapbox(
        geojson=geojson_sulteng,
        locations=[1],
        featureidkey="properties.id",
        color=[1],
        mapbox_style="carto-positron",
        center={"lat": -0.8, "lon": 122.1},
        zoom=6,
        opacity=0.5,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ----------------------------------
    # MODEL PREDIKSI
    # ----------------------------------
    st.subheader("🤖 Prediksi Iklim 10–50 Tahun")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Data minimal harus memiliki ≥2 kolom numerik.")
    else:
        target = st.selectbox("Pilih kolom target (misal: suhu, curah hujan)", numeric_cols)
        fitur = [c for c in numeric_cols if c != target]

        X = df[fitur]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=200)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        st.write("### 🔍 Hasil Prediksi Model")
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": pred})
        st.dataframe(pred_df)

        st.write("### 📈 Prediksi 50 Tahun ke Depan")

        future_years = st.slider("Pilih jumlah tahun prediksi:", 10, 50, 30)

        last_row = df[fitur].iloc[-1:]
        future_pred = model.predict([last_row.values.flatten()])[0]

        st.metric(f"Prediksi Nilai Iklim {future_years} tahun ke depan", f"{future_pred:.2f}")

else:
    st.info("Silakan upload data Excel terlebih dahulu.")


