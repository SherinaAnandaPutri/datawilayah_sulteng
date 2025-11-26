# ================================
# DETEKSI WILAYAH SULAWESI TENGGARA
# ================================

st.subheader("🗺️ Deteksi Wilayah: Sulawesi Tenggara")

# daftar kabupaten/kota Sultra
sultra_regions = [
    "Kendari", "Baubau", "Kolaka", "Kolaka Utara", "Kolaka Timur",
    "Konawe", "Konawe Selatan", "Konawe Kepulauan", "Konawe Utara",
    "Bombana", "Buton", "Buton Utara", "Buton Selatan", 
    "Buton Tengah", "Muna", "Muna Barat", "Wakatobi"
]

# stasiun BMKG di Sultra (contoh)
bmkg_sultra = {
    "Stasiun Meteo Maritim Kendari": (-3.967, 122.600),
    "Stasiun Klimatologi Konawe Selatan": (-4.083, 122.087),
    "Stasiun Geofisika Kendari": (-3.99, 122.51),
    "Stasiun Baubau": (-5.45, 122.62)
}

# fungsi deteksi wilayah
def detect_sultra(df):
    col_names = df.columns.str.lower()

    # 1. deteksi dari kolom nama kota / kabupaten
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].str.contains('|'.join(sultra_regions), case=False, na=False).any():
                return "Sulawesi Tenggara", "nama wilayah"

    # 2. deteksi dari kolom stasiun BMKG
    for st_name in bmkg_sultra.keys():
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].str.contains(st_name, case=False, na=False).any():
                    return "Sulawesi Tenggara", "stasiun BMKG"

    # 3. deteksi dari koordinat
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if lat_cols and lon_cols:
        lat = df[lat_cols[0]].astype(float)
        lon = df[lon_cols[0]].astype(float)

        # range kasar koordinat Sultra
        if (lat.between(-6.2, -3.0).any()) and (lon.between(120.0, 124.0).any()):
            return "Sulawesi Tenggara", "koordinat"

    return "Bukan Sulawesi Tenggara", None


# Jalankan deteksi
wilayah, metode = detect_sultra(df)

if wilayah == "Sulawesi Tenggara":
    st.success(f"✔ Data terdeteksi berasal dari **Wilayah Sulawesi Tenggara** (metode: {metode}).")
else:
    st.warning("⚠ Dataset tidak terdeteksi sebagai wilayah Sulawesi Tenggara. "
               "Jika ini adalah data Sultra, pastikan kolom lokasi/koordinat tersedia.")

# ================================
# TAMPILKAN PETA JIKA SULTRA
# ================================
if wilayah == "Sulawesi Tenggara":
    st.subheader("🗺️ Peta Stasiun BMKG Sulawesi Tenggara")

    st.map(pd.DataFrame({
        'lat': [v[0] for v in bmkg_sultra.values()],
        'lon': [v[1] for v in bmkg_sultra.values()]
    }))

    # statistik cepat
    st.subheader("📌 Statistik Cuaca Utama Wilayah Sulawesi Tenggara")
    st.write(df.describe())
    
    # heatmap korelasi
    st.subheader("🔥 Korelasi Variabel Cuaca (Sulawesi Tenggara)")
    corr_df = df.select_dtypes(include=[np.number]).corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='RdBu',
        title="Heatmap Korelasi Iklim Sulawesi Tenggara"
    )
    st.plotly_chart(fig, use_container_width=True)
