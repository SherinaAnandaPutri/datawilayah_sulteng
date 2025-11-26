import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import openpyxl

# Judul Dashboard
st.title("🌦️ Prediksi Iklim di Wilayah Indonesia dengan Machine Learning")
st.write("Upload data harian untuk melatih model dan memprediksi iklim 10–50 tahun ke depan.")

# Upload Data
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Cek kolom wajib
    required_cols = ["temperature", "humidity", "rainfall"]
    if all(col in df.columns for col in required_cols):

        # Visualisasi Data
        fig = px.line(df[required_cols], title="Tren Variabel Iklim")
        st.plotly_chart(fig)

        # Persiapan Data
        X = df.drop("rainfall", axis=1)
        y = df["rainfall"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        st.subheader("📊 Evaluasi Model")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

        # Input tahun prediksi
        st.subheader("🔮 Prediksi Iklim Masa Depan")
        tahun_prediksi = st.slider("Pilih rentang tahun prediksi", 10, 50, 30)

        # Dummy input masa depan (contoh)
        future_data = pd.DataFrame({
            "temperature": np.linspace(df["temperature"].mean(), df["temperature"].mean()+1, tahun_prediksi),
            "humidity": np.linspace(df["humidity"].mean(), df["humidity"].mean()+1, tahun_prediksi)
        })

        future_pred = model.predict(future_data)
        future_data["predicted_rainfall"] = future_pred

        st.write(future_data)

        # DOWNLOAD EXCEL (Fix)
        def convert_df_to_excel(df):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Prediksi Iklim")
            buffer.seek(0)
            return buffer

        excel_file = convert_df_to_excel(future_data)

        st.download_button(
            label="📥 Download Hasil Prediksi (Excel)",
            data=excel_file,
            file_name="prediksi_iklim.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.error("Kolom wajib tidak lengkap. Harus ada: temperature, humidity, rainfall.")
