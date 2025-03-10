#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("📊 Bankruptcy Prediction App")
st.write("Enter company financial details to predict bankruptcy.")

# User input form
def user_input():
    X1 = st.number_input("X1: Net Profit / Total Assets", value=0.0)
    X2 = st.number_input("X2: Total Liabilities / Total Assets", value=0.0)
    X3 = st.number_input("X3: Working Capital / Total Assets", value=0.0)
    X4 = st.number_input("X4: Retained Earnings / Total Assets", value=0.0)
    X5 = st.number_input("X5: EBIT / Total Assets", value=0.0)
    
    data = pd.DataFrame([[X1, X2, X3, X4, X5]], 
                        columns=["X1", "X2", "X3", "X4", "X5"])
    return data

# Get user input
input_data = user_input()

if st.button("Predict"):
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ Warning! High risk of bankruptcy.")
    else:
        st.success("✅ No bankruptcy risk detected.")


# In[ ]:




