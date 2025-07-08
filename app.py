import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------
# 1. Setup Halaman
# -----------------------------------------------
st.set_page_config(page_title="Prediksi CO2 Emissions", layout="centered")
st.title("🚗 Prediksi Emisi CO₂ Kendaraan")
st.write("Aplikasi ini memprediksi **CO₂ Emissions (g/km)** berdasarkan spesifikasi kendaraan.")

# -----------------------------------------------
# 2. Input User
# -----------------------------------------------
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1, format="%.1f")
cylinders = st.number_input("Jumlah Silinder", min_value=1, max_value=16, step=1)
city = st.number_input("Fuel Consumption City (L/100 km)", min_value=0.0, step=0.1)
hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0, step=0.1)
comb = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=0.0, step=0.1)

if st.button("Prediksi Emisi CO₂"):
    # -----------------------------------------------
    # 3. Proses Data dan Model
    # -----------------------------------------------

    # Fitur dan target yang digunakan
    features = [
        "Engine Size(L)",
        "Cylinders",
        "Fuel Consumption City (L/100 km)",
        "Fuel Consumption Hwy (L/100 km)",
        "Fuel Consumption Comb (L/100 km)"
    ]
    target = "CO2 Emissions(g/km)"

    # Load dataset dari GitHub
    url = "https://raw.githubusercontent.com/RazeYuki/EMISI/main/Fuel%20Consumption%20Ratings.csv"
    data = pd.read_csv(url)

    # Hapus duplikat dan bersihkan
    data = data.drop_duplicates()
    data.columns = data.columns.str.strip()

    # Ambil X dan y
    X = data[features]
    y = data[target]

    # Normalisasi dengan MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Latih model Linear Regression
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Transform input user
    user_input = np.array([[engine_size, cylinders, city, hwy, comb]])
    user_input_scaled = scaler.transform(user_input)

    # Prediksi
    prediction = model.predict(user_input_scaled)[0]

    # -----------------------------------------------
    # 4. Tampilkan Hasil
    # -----------------------------------------------
    st.success(f"Perkiraan CO₂ Emissions: **{prediction:.2f} g/km**")
