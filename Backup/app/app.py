# app/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from src.predict import predict_campaign_response

st.set_page_config(page_title="Marketing Campaign Optimizer", layout="centered")

st.title("📣 Marketing Campaign Optimizer")
st.markdown("Predict whether a customer will subscribe to your campaign.")

# --- Form Input ---
with st.form("user_input_form"):
    st.subheader("Enter Customer Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
    ])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("Default Credit?", ['yes', 'no'])
    balance = st.number_input("Account Balance", value=0)
    housing = st.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.selectbox("Personal Loan", ['yes', 'no'])
    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    day = st.slider("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    duration = st.slider("Call Duration (seconds)", min_value=0, max_value=3000, value=200)
    campaign = st.number_input("No. of Contacts in Campaign", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact (-1 if never)", value=-1)
    previous = st.number_input("Previous Contacts", value=0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ['success', 'failure', 'other', 'unknown'])

    submit = st.form_submit_button("Predict")

# --- Make Prediction ---
if submit:
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    result = predict_campaign_response(input_dict)

    st.subheader("📊 Prediction Result:")
    if result["prediction"] == 1:
        st.success(f"✅ Likely to Subscribe! (Probability: {result['probability_of_yes']})")
    else:
        st.error(f"❌ Not Likely to Subscribe (Probability: {result['probability_of_yes']})")
