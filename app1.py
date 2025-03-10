import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model & scaler
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define app title
st.title("📉 Bankruptcy Prediction App")
st.write("Enter company financial data to predict bankruptcy.")

# Input fields
industrial_risk = st.number_input("Industrial Risk", min_value=0.0, max_value=1.0, value=0.5)
management_risk = st.number_input("Management Risk", min_value=0.0, max_value=1.0, value=0.5)
financial_flexibility = st.number_input("Financial Flexibility", min_value=0.0, max_value=1.0, value=0.5)
credibility = st.number_input("Credibility", min_value=0.0, max_value=1.0, value=0.5)
competitiveness = st.number_input("Competitiveness", min_value=0.0, max_value=1.0, value=0.5)
operating_risk = st.number_input("Operating Risk", min_value=0.0, max_value=1.0, value=0.5)

# Collect user input
user_input = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])

# Scale input
user_input_scaled = scaler.transform(user_input)

# Predict on user input
if st.button("🔮 Predict"):
    prediction = model.predict(user_input_scaled)
    result = "❌ Bankrupt" if prediction[0] == 1 else "✅ Not Bankrupt"
    st.subheader(f"🔍 Prediction: {result}")
