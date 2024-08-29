import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model(r"C:\Users\91978\Downloads\mode.h5")
scaler = joblib.load(r"C:\Users\91978\Downloads\scale.pkl")

# Streamlit app title
st.title("Telecom Churn Prediction")

# Input fields for all 13 features
age = st.number_input("Age", min_value=0)
subscription_length = st.number_input("Subscription Length (months)", min_value=1, max_value=100)
charge_amount = st.selectbox("Charge Amount (0: lowest, 9: highest)", options=range(10))
seconds_of_use = st.number_input("Total Seconds of Use", min_value=0)
frequency_of_use = st.number_input("Total Number of Calls", min_value=0)
frequency_of_sms = st.number_input("Total Number of SMS", min_value=0)
distinct_called_numbers = st.number_input("Total Distinct Called Numbers", min_value=0)
age_group = st.selectbox("Age Group (1: youngest, 5: oldest)", options=range(1, 6))
tariff_plan = st.selectbox("Tariff Plan (1: Pay as you go, 2: Contractual)", options=[1, 2])
status = st.selectbox("Status (1: Active, 2: Non-active)", options=[1, 2])
call_failures = st.number_input("Number of Call Failures", min_value=0)
complains = st.selectbox("Complaints (0: No, 1: Yes)", options=[0, 1])
customer_value = st.number_input("Customer Value", min_value=0)

if st.button("Predict"):
    # Create an input array with the 13 features
    input_data = np.array([[age, subscription_length, charge_amount, seconds_of_use,
                            frequency_of_use, frequency_of_sms, distinct_called_numbers,
                            age_group, tariff_plan, status, call_failures, complains,
                            customer_value]])  # 13 features
    
    # Scale the input data using the pre-fitted scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction result
    if prediction > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
