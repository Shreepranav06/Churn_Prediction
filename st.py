import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
from tensorflow.keras.models import load_model
import os

# URLs of the model and scaler
model_url = 'https://github.com/Shreepranav06/Churn_Prediction/raw/main/mode.h5'
scaler_url = 'https://github.com/Shreepranav06/Churn_Prediction/raw/main/scale.pkl'

# Paths to save the downloaded files
model_path = 'model.h5'
scaler_path = 'scaler.pkl'

# Function to download files
def download_file(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        st.error(f"Failed to download file from {url}")
        st.stop()

# Download model and scaler if not already downloaded
if not os.path.exists(model_path):
    download_file(model_url, model_path)
if not os.path.exists(scaler_path):
    download_file(scaler_url, scaler_path)

# Load the model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

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
