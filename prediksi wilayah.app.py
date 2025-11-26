# premium_dashboard.py
"""
Premium dashboard: auto-detect features, multi-model training with RandomizedSearchCV,
export to Excel multi-sheet (data, train/test, predictions, evaluation) and download model (.pkl).
Designed to be robust and run on Streamlit Cloud (no heavy external deps).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Premium — Prediksi Iklim (SULTENG/Kendari)", layout="wide")
st.title("🚀 Premium: Prediksi Iklim — Auto ML & Export (Excel + Model)")
st.markdown("Fitur: auto-detect, hyperparameter tuning (RandomizedSearchCV), multiple algorithms, export Excel multi-sheet, download model pickle.")

# -------------------------
# Helper utilities
# -------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def eval_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

def df_to_excel_bytes(sheets_dict):
    """
    sheets_dict: {sheet_name: dataframe or (image_bytes, df?)}
    We'll write dataframes to sheets. Also allow embedding png images (matplotlib) into first sheet.
    For simplicity, we write dataframes using pd.ExcelWriter (xlsxwriter/openpyxl).
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets_dict.items():
            # ensure sheet name <=31 chars
            sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
        writer.save()
    output.seek(0)
    return output.getvalue()

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def download_file(data_bytes, filename, mime):
    st.download_button(label=f"📥 Download {filename}", data=data_bytes, file_name=filename, mime=mime)

# -------------------------
# Sample data (if user does not upload)
# monthly timeseries example
# -------------------------
def make_sample_monthly(year_start=2000, year_end=2020, seed=42):
    np.random.seed(seed)
    years = np.arange(year_start, year_end+1)
    rows = []
    for y in years:
        for m in range(1,13):
            rows.append({
                "tahun": y,
                "bulan": m,
                "Tn": np.random.uniform(22,25),
                "Tx": np.random.uniform(29,33),
                "Tavg": np.random.uniform(25,28),
                "kelembaban": np.random.uniform(60,90),
                "curah_hujan": np.random.uniform(0,350),
                "matahari": np.random.uniform(0,10),
            })
    return pd.DataFrame(rows)

# -------------------------
# Sidebar: data upload or sample
# -------------------------
st.sidebar.header("Data & Mode")
data_mode = st.sidebar.radio("Pilih sumber data", ("Gunakan data contoh (sample)", "Upload file (CSV/XLSX)"))

uploaded_df = None
if data_mode.startswith("Upload"):
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau XLSX", type=["csv","xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
            st.sidebar.success("File berhasil dibaca.")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")
            uploaded_df = None

if data_mode.startswith("Gunakan"):
    st.sidebar.info("Menggunakan dataset contoh (monthly) untuk demo.")
    df = make_sample_monthly(2000, 2020)
else:
    if uploaded_df is None:
        st.warning("Belum ada file diupload — gunakan sample atau upload file.")
        df = make_sample_monthly(2000, 2020)
    else:
        df = uploaded_df.copy()

st.write("### Preview data (otomatis deteksi kolom numeric)")
st.dataframe(df.head())

# -------------------------
# Auto-detect numeric columns and let user pick target
# -------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik di dataset — pastikan file berisi kolom numerik untuk dipakai sebagai fitur/target.")
    st.stop()

st.sidebar.subheader("Feature selection")
st.sidebar.write("Kolom numerik yang terdeteksi:")
st.sidebar.write(numeric_cols)

# target selection
target_col = st.sidebar.selectbox("Pilih kolom target (yang ingin diprediksi)", numeric_cols, index=len(numeric_cols)-1)
feature_candidates = [c for c in numeric_cols if c != target_col]
selected_features = st.sidebar.multiselect("Pilih fitur (default: semua numeric selain target)", feature_candidates, default=feature_candidates)

if len(selected_features) == 0:
    st.error("Pilih minimal 1 fitur untuk model.")
    st.stop()

# preprocessing options
st.sidebar.subheader("Preprocessing")
do_scale = st.sidebar.checkbox("Standard scaling (fit pada training)", value=True)
impute_strategy = st.sidebar.selectbox("Imputer strategy", ("mean", "median", "most_frequent"), index=0)

# modeling options
st.sidebar.subheader("Modeling & Tuning")
model_choices = st.sidebar.multiselect("Pilih algoritma", ["RandomForest", "GradientBoosting", "MLP"], default=["RandomForest","GradientBoosting"])
n_iter = st.sidebar.slider("RandomizedSearch n_iter (hyperparam tuning)", min_value=5, max_value=80, value=20, step=5)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# prediction options
st.sidebar.subheader("Prediksi")
horizon_years = st.sidebar.slider("Horizon prediksi (tahun)", 10, 100, 30, step=5)
frequency = st.sidebar.selectbox("Frekuensi prediksi", ["yearly", "monthly"], index=0)

# train/test split
test_size = st.sidebar.slider("Proporsi test set", 0.1, 0.4, 0.2, step=0.05)

# run button
run_button = st.button("🔁 Jalankan training & tuning")

# -------------------------
# If run: prepare data, pipelines, param grids, RandomizedSearchCV
# -------------------------
if run_button:
    st.info("Memulai proses training. Ini mungkin membutuhkan beberapa detik hingga beberapa menit tergantung n_iter.")
    X = df[selected_features].copy()
    y = df[target_col].copy()

    # simple imputing
    imputer = SimpleImputer(strategy=impute_strategy)
    try:
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    except Exception as e:
        st.error(f"Gagal melakukan imputasi: {e}")
        st.stop()

    # split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        st.error(f"Gagal split data: {e}")
        st.stop()

    # pipeline building helper
    def build_pipeline(estimator, scale=do_scale):
        steps = []
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("est", estimator))
        return Pipeline(steps)

    # models and parameter spaces
    models_and_params = {}

    if "RandomForest" in model_choices:
        models_and_params["RandomForest"] = {
            "pipe": build_pipeline(RandomForestRegressor(random_state=random_state)),
            "params": {
                "est__n_estimators": [100,200,300,500],
                "est__max_depth": [None, 5, 10, 20],
                "est__min_samples_split": [2,5,10],
                "est__min_samples_leaf": [1,2,4]
            }
        }
    if "GradientBoosting" in model_choices:
        models_and_params["GradientBoosting"] = {
            "pipe": build_pipeline(GradientBoostingRegressor(random_state=random_state)),
            "params": {
                "est__n_estimators": [100,200,300],
                "est__learning_rate": [0.01,0.05,0.1],
                "est__max_depth": [3,5,8],
                "est__subsample": [0.6,0.8,1.0]
            }
        }
    if "MLP" in model_choices:
        models_and_params["MLP"] = {
            "pipe": build_pipeline(MLPRegressor(random_state=random_state, max_iter=1000)),
            "params": {
                "est__hidden_layer_sizes": [(50,),(100,),(100,50)],
                "est__alpha": [0.0001, 0.001, 0.01],
                "est__learning_rate_init": [0.001,0.01]
            }
        }

    results = []
    best_models = {}
    for name, mp in models_and_params.items():
        st.write(f"▶️ Tuning model: **{name}**")
        pipe = mp["pipe"]
        param_dist = mp["params"]

        try:
            rand = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=3, random_state=random_state, n_jobs=-1)
            rand.fit(X_train, y_train)
            best = rand.best_estimator_
            # predict
            y_pred_train = best.predict(X_train)
            y_pred_test = best.predict(X_test)
            metrics_train = eval_metrics(y_train, y_pred_train)
            metrics_test = eval_metrics(y_test, y_pred_test)
            results.append({
                "model": name,
                "best_params": rand.best_params_,
                "train_metrics": metrics_train,
                "test_metrics": metrics_test
            })
            best_models[name] = best
            st.success(f"{name} selesai. Test R2 = {metrics_test['R2']:.4f}")
        except Exception as e:
            st.error(f"Gagal tuning model {name}: {e}")
            continue

    if len(best_models) == 0:
        st.error("Tidak ada model yang berhasil dituning.")
    else:
        # show summary table
        st.subheader("Ringkasan Hasil Tuning")
        rows = []
        for r in results:
            rows.append({
                "model": r["model"],
                "test_R2": r["test_metrics"]["R2"],
                "test_RMSE": r["test_metrics"]["RMSE"],
                "test_MAE": r["test_metrics"]["MAE"],
                "best_params": str(r["best_params"])
            })
        df_summary = pd.DataFrame(rows).sort_values("test_R2", ascending=False).reset_index(drop=True)
        st.dataframe(df_summary)

        # pick best model by test R2
        best_name = df_summary.iloc[0]["model"]
        st.success(f"Model terbaik berdasarkan R² test: {best_name}")
        best_model = best_models[best_name]

        # save model to bytes
        model_bytes = io.BytesIO()
        joblib.dump(best_model, model_bytes)
        model_bytes.seek(0)

        # allow download model
        download_file(model_bytes.getvalue(), f"model_best_{best_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl", "application/octet-stream")

        # Create predictions for horizon
        st.subheader("Prediksi Masa Depan")
        if frequency == "yearly":
            # create yearly feature rows using mean month or just month=1
            last_year = int(df["tahun"].max()) if "tahun" in df.columns else datetime.now().year
            years = np.arange(last_year+1, last_year+1+horizon_years)
            # for features, use 'tahun' and set month to 1 (or median month)
            rows = []
            for y in years:
                # create row using median/mode for non-year features if present
                base = {}
                if "tahun" in selected_features:
                    base["tahun"] = y
                if "bulan" in selected_features:
                    base["bulan"] = 1
                # for any other feature, use column mean from df
                for feat in selected_features:
                    if feat not in ["tahun","bulan"]:
                        base[feat] = df[feat].mean() if feat in df.columns else 0.0
                rows.append(base)
            X_future = pd.DataFrame(rows)[selected_features]
            preds = best_model.predict(X_future)
            X_future["predicted"] = preds
            st.write(X_future.head())
        else:  # monthly
            last_year = int(df["tahun"].max()) if "tahun" in df.columns else datetime.now().year
            months = horizon_years * 12
            rows = []
            for i in range(1, months+1):
                yval = last_year + (i // 12) + 1 if i%12==0 else last_year + (i // 12)
                mval = (i % 12) if (i % 12) != 0 else 12
                base = {}
                if "tahun" in selected_features:
                    base["tahun"] = yval
                if "bulan" in selected_features:
                    base["bulan"] = mval
                for feat in selected_features:
                    if feat not in ["tahun","bulan"]:
                        base[feat] = df[feat].mean() if feat in df.columns else 0.0
                rows.append(base)
            X_future = pd.DataFrame(rows)[selected_features]
            preds = best_model.predict(X_future)
            X_future["predicted"] = preds
            st.write(X_future.head())

        # Evaluate best model on test set again and show metrics
        y_test_pred = best_model.predict(X_test)
        met = eval_metrics(y_test, y_test_pred)
        st.subheader("Evaluasi model terbaik pada test set")
        st.write(met)

        # Plot actual vs predicted (test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {best_name}")
        st.pyplot(fig)

        # Prepare Excel multi-sheet
        st.subheader("Export: Excel multi-sheet & download")
        # Data sheets: original data, train set, test set, future predictions, eval
        try:
            df_train = pd.DataFrame(X_train, columns=selected_features)
            df_train[target_col] = y_train.values
            df_test = pd.DataFrame(X_test, columns=selected_features)
            df_test[target_col] = y_test.values
            df_future = X_future.copy()
            df_future["predicted"] = preds
            df_eval = pd.DataFrame([met], index=[best_name])

            sheets = {
                "original_data": df.reset_index(drop=True),
                "train_set": df_train.reset_index(drop=True),
                "test_set": df_test.reset_index(drop=True),
                "predictions": df_future.reset_index(drop=True),
                "evaluation": df_eval.reset_index()
            }

            excel_bytes = df_to_excel_bytes(sheets)
            download_file(excel_bytes, f"prediksi_report_{best_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.success("File Excel siap diunduh.")
        except Exception as e:
            st.error(f"Gagal membuat file Excel: {e}")
