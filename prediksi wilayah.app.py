# dashboard_kendari_upgraded.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io, base64

# -------------------------
# PAGE CONFIG & CSS
# -------------------------
st.set_page_config(page_title="Dashboard Iklim Kendari — Upgraded", layout="wide")
st.markdown("""
<style>
/* Simple pleasant theme */
body {background: linear-gradient(180deg,#f7fbff 0%,#eef6ff 100%);}
.header {color:#0b5ed7; font-weight:700;}
.card {background:white; padding:12px; border-radius:8px; box-shadow:0 6px 18px rgba(11,94,215,0.06);}
.small {font-size:0.9rem; color:#666;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header'>🌦️ Dashboard Iklim — Kota Kendari (Upgraded)</h1>", unsafe_allow_html=True)
st.write("Fitur: upload data (opsional), peta heatmap, grafik tren, prediksi 2025–2075, evaluasi & download.")

# -------------------------
# Kecamatan + lokasi (lat, lon)
# -------------------------
kecamatan_locs = {
    "Mandonga": (-3.967, 122.514),
    "Baruga": (-4.005, 122.495),
    "Kadia": (-3.996, 122.529),
    "Wua-Wua": (-4.003, 122.523),
    "Poasia": (-4.030, 122.542),
    "Kambu": (-4.020, 122.509),
}

kecamatans = list(kecamatan_locs.keys())

# -------------------------
# Helper: make sample timeseries per kecamatan (2010-2024 monthly)
# -------------------------
def make_sample_monthly(seed):
    np.random.seed(seed)
    records = []
    years = range(2010, 2025)
    for y in years:
        for m in range(1,13):
            records.append({
                "tahun": y,
                "bulan": m,
                "Tn": round(np.random.uniform(22,25),2),
                "Tx": round(np.random.uniform(29,33),2),
                "Tavg": round(np.random.uniform(25,28),2),
                "Kelembaban": round(np.random.uniform(60,90),1),
                "Curah_Hujan": round(np.random.uniform(20,300),1),
                "Matahari": round(np.random.uniform(0,10),2),
                "Angin": round(np.random.uniform(0.5,6),2)
            })
    df = pd.DataFrame(records)
    return df

# default data per kecamatan
data_by_kec = {k: make_sample_monthly(i+10) for i,k in enumerate(kecamatans)}

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Kontrol")
use_upload = st.sidebar.checkbox("Upload data per-kecamatan (opsional)", value=False)
upload_kec = st.sidebar.selectbox("Pilih kecamatan untuk upload (jika dipilih)", kecamatans)

uploaded = None
if use_upload:
    uploaded = st.sidebar.file_uploader(f"Upload CSV/XLSX untuk {upload_kec} (kolom: tahun,bulan,Tn,Tx,Tavg,Kelembaban,Curah_Hujan,Matahari,Angin)", type=["csv","xlsx"], key="upload1")
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            # normalize column names
            df_up.columns = [c.strip() for c in df_up.columns]
            # try to detect / rename common alternative names
            colmap = {}
            for c in df_up.columns:
                lc = c.lower()
                if "tahun" in lc: colmap[c] = "tahun"
                if lc in ["month","bulan"]: colmap[c] = "bulan"
                if "tn" in lc or "min" in lc and "temp" in lc: colmap[c] = "Tn"
                if "tx" in lc or "max" in lc and "temp" in lc: colmap[c] = "Tx"
                if "tavg" in lc or "avg" in lc: colmap[c] = "Tavg"
                if "kelembab" in lc: colmap[c] = "Kelembaban"
                if "hujan" in lc or "rain" in lc: colmap[c] = "Curah_Hujan"
                if "mata" in lc or "sun" in lc: colmap[c] = "Matahari"
                if "angin" in lc or "wind" in lc: colmap[c] = "Angin"
            df_up = df_up.rename(columns=colmap)
            needed = ["tahun","bulan","Tn","Tx","Tavg","Kelembaban","Curah_Hujan","Matahari","Angin"]
            if not all(x in df_up.columns for x in needed):
                st.sidebar.error("File upload tidak lengkap. Pastikan mengandung kolom: tahun, bulan, Tn, Tx, Tavg, Kelembaban, Curah_Hujan, Matahari, Angin")
            else:
                df_up = df_up[needed].copy()
                df_up = df_up.sort_values(["tahun","bulan"]).reset_index(drop=True)
                data_by_kec[upload_kec] = df_up
                st.sidebar.success(f"Data {upload_kec} berhasil dimuat ({len(df_up)} baris).")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")

# -------------------------
# Main layout: top metrics
# -------------------------
st.subheader("Ringkasan Data & Pilihan Visualisasi")
col1, col2, col3 = st.columns(3)
total_records = sum(len(df) for df in data_by_kec.values())
col1.metric("Total record (semua kec.)", f"{total_records:,}")
col2.metric("Kecamatan", f"{len(kecamatans)}")
col3.metric("Periode sample", "2010–2024 (monthly sample)")

# -------------------------
# Map section (folium markers + circle markers as heat)
# -------------------------
st.header("🗺️ Peta — Persebaran & Heat (marker)")

map_center = (-3.99, 122.52)
m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")

# choose metric for map
map_metric = st.selectbox("Pilih metrik untuk peta (menggunakan nilai tahun terakhir tiap kecamatan)", ["Tavg","Curah_Hujan","Kelembaban","Angin"], index=0)

# prepare last-year values per kecamatan
last_values = {}
for k, dfk in data_by_kec.items():
    try:
        last = dfk.sort_values(["tahun","bulan"]).iloc[-1]
        last_values[k] = float(last[map_metric])
    except Exception:
        last_values[k] = np.nan

# compute color scale
vals = np.array([v for v in last_values.values() if not np.isnan(v)])
vmin = float(vals.min()) if len(vals)>0 else 0.0
vmax = float(vals.max()) if len(vals)>0 else 1.0

for k, (lat, lon) in kecamatan_locs.items():
    val = last_values.get(k, np.nan)
    # color mapping (simple)
    if np.isnan(val):
        color = "gray"
        radius = 6
    else:
        # normalize 0-1
        norm = (val - vmin) / (vmax - vmin) if vmax>vmin else 0.5
        # color gradient blue->red
        r = int(255 * norm)
        b = int(255 * (1-norm))
        color = f"#{r:02x}00{b:02x}"
        radius = 6 + int(14 * norm)
    folium.CircleMarker(location=(lat, lon),
                        radius=radius,
                        color=color,
                        fill=True, fill_opacity=0.7,
                        popup=f"{k}: {map_metric} = {val:.2f}" if not np.isnan(val) else f"{k}: N/A").add_to(m)
    folium.Marker(location=(lat, lon), tooltip=k, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

st_folium(m, height=480, width=900)

st.caption("Marker = lokasi kecamatan. Circle size & color merepresentasikan nilai metrik (tahun terakhir).")

# -------------------------
# Trends & variable selection
# -------------------------
st.header("📈 Grafik Tren & Analisis")

selected_kec = st.selectbox("Pilih kecamatan untuk analisis tren", kecamatans)
df_sel = data_by_kec[selected_kec].copy()
df_sel["Tanggal"] = pd.to_datetime(df_sel["tahun"].astype(str) + "-" + df_sel["bulan"].astype(str) + "-01")

var_plot = st.multiselect("Pilih variabel untuk plot (multi)", ["Tn","Tx","Tavg","Kelembaban","Curah_Hujan","Matahari","Angin"],
                          default=["Tavg","Curah_Hujan"])

try:
    fig_trend = px.line(df_sel, x="Tanggal", y=var_plot, labels={"value":"Nilai","variable":"Variabel"}, title=f"Tren: {', '.join(var_plot)} — {selected_kec}")
    st.plotly_chart(fig_trend, use_container_width=True)
except Exception as e:
    st.error(f"Gagal menggambar grafik tren: {e}")

# -------------------------
# Modeling & Predictions
# -------------------------
st.header("🔮 Prediksi 2025–2075 (per kecamatan)")

predict_var = st.selectbox("Pilih variabel target prediksi", ["Tn","Tx","Tavg","Curah_Hujan","Kelembaban","Matahari","Angin"], index=2)
n_estimators = st.slider("Jumlah pohon RandomForest", 50, 300, 150, step=50)
test_size = st.slider("Proporsi test untuk evaluasi", 0.05, 0.4, 0.2, step=0.05)
run_pred = st.button("🔁 Jalankan Prediksi untuk semua kecamatan")

df_all_preds = None
metrics_list = []

if run_pred:
    all_preds = []
    for k in kecamatans:
        try:
            dfi = data_by_kec[k].copy()
            # prepare features: year & month as numeric
            X = dfi[["tahun","bulan"]].values
            y = dfi[predict_var].values
            if len(dfi) < 8:
                st.warning(f"{k}: data kurang (<8), prediksi di-skip.")
                continue
            Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=test_size, random_state=42)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(Xtr, ytr)
            ypred_test = model.predict(Xts)
            rmse = np.sqrt(mean_squared_error(yts, ypred_test))
            mae = mean_absolute_error(yts, ypred_test)
            r2 = r2_score(yts, ypred_test) if len(np.unique(yts))>1 else float("nan")
            metrics_list.append({"kecamatan":k,"rmse":rmse,"mae":mae,"r2":r2})
            # predict monthly 2025-2075
            years = np.arange(2025, 2076)
            months = np.arange(1,13)
            rows = []
            for yy in years:
                for mm in months:
                    rows.append([yy, mm])
            X_future = np.array(rows)
            y_future = model.predict(X_future)
            dfp = pd.DataFrame(X_future, columns=["tahun","bulan"])
            dfp[f"pred_{predict_var}"] = y_future
            dfp["kecamatan"] = k
            all_preds.append(dfp)
        except Exception as e:
            st.error(f"Gagal prediksi {k}: {e}")
            continue
    if len(all_preds)>0:
        df_all_preds = pd.concat(all_preds, ignore_index=True)
        st.success("Prediksi selesai.")
        st.subheader("Metrik Evaluasi (test split)")
        st.dataframe(pd.DataFrame(metrics_list).set_index("kecamatan"))
        st.subheader("Contoh: prediksi 2025–2075 (pilih kecamatan untuk tabel/grafik)")
        pick = st.selectbox("Pilih kecamatan untuk lihat prediksi", kecamatans, key="pick_pred")
        st.dataframe(df_all_preds[df_all_preds["kecamatan"]==pick].head(24), use_container_width=True)
        # plot yearly aggregated
        df_plot = df_all_preds.copy()
        df_plot["tahun"] = df_plot["tahun"].astype(int)
        yearly = df_plot.groupby(["kecamatan","tahun"]).mean().reset_index()
        figp = px.line(yearly[yearly["kecamatan"]==pick], x="tahun", y=f"pred_{predict_var}", title=f"Prediksi tahunan {predict_var} — {pick}")
        st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("Tidak ada prediksi dibuat (data kurang atau error).")

# -------------------------
# Download predictions
# -------------------------
st.header("📥 Download Hasil Prediksi")
if df_all_preds is None:
    st.info("Jalankan prediksi dulu untuk mengaktifkan tombol download.")
else:
    csv = df_all_preds.to_csv(index=False)
    st.download_button("Download CSV - prediksi semua kecamatan", data=csv, file_name="prediksi_kendari_2025_2075.csv", mime="text/csv")

    # Excel multi-sheet
    try:
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            for k in kecamatans:
                dfx = df_all_preds[df_all_preds["kecamatan"]==k].reset_index(drop=True)
                dfx.to_excel(writer, sheet_name=k[:31], index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediksi_kendari_2025_2075.xlsx">Download Excel (per-kecamatan)</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal membuat file Excel: {e}")

st.markdown("---")
st.caption("Versi upgraded: peta marker+heat, prediksi monthly 2025–2075, upload opsional. Untuk peta polygon akurat, sediakan GeoJSON resmi.")
