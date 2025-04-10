import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("svm_stock_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 Stock Close Price Prediction")

# Input fields
open_val = st.number_input("Open Price", value=80.0)
high_val = st.number_input("High Price", value=81.0)
low_val = st.number_input("Low Price", value=79.5)
volume_val = st.number_input("Volume", value=1000000)

if st.button("Predict Close Price"):
    # Prepare input
    input_data = np.array([[open_val, high_val, low_val, volume_val]])
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"📊 Predicted Close Price: **${prediction:.2f}**")
