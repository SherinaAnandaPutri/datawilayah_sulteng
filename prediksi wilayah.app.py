import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# 1. DATA DEFAULT SULTENG(1)
# ============================================================

df_default = pd.DataFrame({
    "tahun": np.arange(2000, 2021),
    "suhu": np.random.uniform(25, 29, 21),
    "hujan": np.random.uniform(1200, 2500, 21),
    "lembap": np.random.uniform(65, 90, 21)
})

# ============================================================
# 2. GEOJSON SULTENG(1) TANPA FILE
# ============================================================

geojson_sulteng = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "SULTENG(1)"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [119.80, -1.0],
                [121.00, -1.0],
                [121.00, -2.0],
                [119.80, -2.0],
                [119.80, -1.0]
            ]]
        }
    }]
}

# ============================================================
# 3. UI TEMA BIRU
# ============================================================

st.set_page_config(page_title="Dashboard Iklim SULTENG(1)", layout="wide")

st.markdown("""
    <h1 style='text-align:center;color:#0066CC;'>🌦️ Dashboard Iklim — SULTENG(1)</h1>
    <p style='text-align:center;font-size:18px;'>
        Analisis tren iklim, prediksi jangka panjang, dan visualisasi wilayah.
    </p>
""", unsafe_allow_html=True)

# ============================================================
# 4. OPSIONAL — UPLOAD DATA
# ============================================================

uploaded = st.file_uploader("Upload data Excel (opsional)", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    st.success("Berhasil memuat data upload!")
else:
    df = df_default
    st.info("Menggunakan data default SULTENG(1).")

# ============================================================
# 5. TAMPILKAN DATA
# ============================================================

st.subheader("📊 Data Iklim")
st.dataframe(df)

# ============================================================
# 6. GRAFIK TREN
# ============================================================

st.subheader("📈 Grafik Tren Iklim")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Suhu")
    fig_suhu = px.line(df, x="tahun", y="suhu")
    st.plotly_chart(fig_suhu, use_container_width=True)

with col2:
    st.markdown("### Curah Hujan")
    fig_hujan = px.line(df, x="tahun", y="hujan")
    st.plotly_chart(fig_hujan, use_container_width=True)

with col3:
    st.markdown("### Kelembapan")
    fig_lembap = px.line(df, x="tahun", y="lembap")
    st.plotly_chart(fig_lembap, use_container_width=True)

# ============================================================
# 7. MODEL PREDIKSI ML
# ============================================================

st.subheader("🤖 Prediksi Iklim 2021–2070 (Random Forest)")

model = RandomForestRegressor(n_estimators=300, random_state=42)

X = df[["tahun"]]
y = df["suhu"]

model.fit(X, y)

tahun_prediksi = np.arange(2021, 2071)
prediksi = model.predict(tahun_prediksi.reshape(-1, 1))

df_prediksi = pd.DataFrame({
    "tahun": tahun_prediksi,
    "prediksi_suhu": prediksi
})

# ============================================================
# 8. GRAFIK PREDIKSI
# ============================================================

fig_pred = px.line(df_prediksi, x="tahun", y="prediksi_suhu",
                   title="Prediksi Suhu 2021–2070")
st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# 9. METRIK AKURASI
# ============================================================

y_pred_train = model.predict(X)

colA, colB, colC = st.columns(3)

with colA:
    st.metric("RMSE", round(np.sqrt(mean_squared_error(y, y_pred_train)), 3))

with colB:
    st.metric("MAE", round(mean_absolute_error(y, y_pred_train), 3))

with colC:
    st.metric("R² Score", round(r2_score(y, y_pred_train), 3))

# ============================================================
# 10. FITUR DOWNLOAD
# ============================================================

st.subheader("📥 Download Hasil Prediksi")

df_download = df_prediksi.copy()

st.download_button(
    label="Download Excel (.xlsx)",
    data=df_download.to_csv(index=False).encode(),
    file_name="prediksi_sulteng1.csv",
    mime="text/csv"
)

# ============================================================
# 11. PETA WILAYAH
# ============================================================

st.subheader("🗺️ Peta Wilayah SULTENG(1)")

fig_map = px.choropleth_mapbox(
    geojson=geojson_sulteng,
    locations=["SULTENG(1)"],
    featureidkey="properties.name",
    center={"lat": -1.5, "lon": 120.3},
    mapbox_style="carto-positron",
    zoom=5,
    color_discrete_sequence=["blue"]
)

st.plotly_chart(fig_map, use_container_width=True)
