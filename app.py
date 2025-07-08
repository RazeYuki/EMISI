import streamlit as st
import numpy as np
import pickle

# ------------------------------
# Load Model dan Scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------------
# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Emisi CO₂ Kendaraan", layout="centered")
st.title("🚗 Prediksi Emisi Karbon Dioksida (CO₂)")
st.markdown("Pilih spesifikasi kendaraan Anda untuk memprediksi **emisi CO₂ (gram/km)**.")

# ------------------------------
# Input User dengan SLIDER
engine_size = st.slider("🔧 Ukuran Mesin (Liter)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
cylinders = st.slider("⚙️ Jumlah Silinder", min_value=2, max_value=16, value=4, step=1)
fuel_city = st.slider("🏙️ Konsumsi BBM Kota (L/100 km)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
fuel_hwy = st.slider("🛣️ Konsumsi BBM Tol (L/100 km)", min_value=1.0, max_value=25.0, value=7.0, step=0.1)
fuel_comb = st.slider("🔄 Konsumsi BBM Kombinasi (L/100 km)", min_value=1.0, max_value=28.0, value=8.5, step=0.1)

# ------------------------------
# Prediksi
if st.button("🔍 Prediksi Emisi CO₂"):
    # Susun data input
    input_data = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    # Tampilkan hasil
    st.success(f"🌿 Estimasi Emisi CO₂ Kendaraan: **{prediction:.2f} gram/km**")
