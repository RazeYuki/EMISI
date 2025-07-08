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
st.markdown("Masukkan spesifikasi kendaraan Anda untuk memprediksi **emisi CO₂ (gram/km)**.")

# ------------------------------
# Input User (Fitur dengan nama Indonesia)
engine_size = st.number_input("🔧 Ukuran Mesin (Liter)", min_value=0.0, step=0.1, format="%.1f")
cylinders = st.number_input("⚙️ Jumlah Silinder", min_value=1, step=1)
fuel_city = st.number_input("🏙️ Konsumsi BBM Kota (L/100 km)", min_value=0.0, step=0.1,
                            help="Fuel Consumption (City)")
fuel_hwy = st.number_input("🛣️ Konsumsi BBM Tol (L/100 km)", min_value=0.0, step=0.1,
                           help="Fuel Consumption (Hwy)")
fuel_comb = st.number_input("🔄 Konsumsi BBM Kombinasi (L/100 km)", min_value=0.0, step=0.1,
                            help="Fuel Consumption (Comb)")

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
