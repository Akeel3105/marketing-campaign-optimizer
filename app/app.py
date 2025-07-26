# app/app.py

import streamlit as st
import pandas as pd
import os
import sys

# Fix path to access src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_campaign_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(page_title="Marketing Campaign Optimizer", layout="wide")

st.title("📣 Marketing Campaign Optimizer")
st.markdown("Predict whether a customer will subscribe to your marketing campaign.")

# ============ Sidebar KPIs ============ #
with st.sidebar:
    st.header("📊 Campaign Insights")

    if os.path.exists("logs/single_predictions_log.csv"):
        log_df = pd.read_csv("logs/single_predictions_log.csv")

        total = len(log_df)
        subscribed = log_df['prediction'].sum()
        not_subscribed = total - subscribed
        avg_prob = log_df['probability_of_yes'].mean()

        st.metric("Total Leads Tested", total)
        st.metric("Subscribed", subscribed)
        st.metric("Not Subscribed", not_subscribed)
        st.metric("Avg. Subscription Probability", round(avg_prob, 2))

        # Optional: Filter by Job
        job_filter = st.selectbox("Filter by Job", options=['All'] + log_df['job'].dropna().unique().tolist())
        if job_filter != 'All':
            filtered_df = log_df[log_df['job'] == job_filter]
        else:
            filtered_df = log_df

        st.bar_chart(filtered_df['prediction'].value_counts())

    else:
        st.info("No predictions yet. Submit data to populate insights.")

# ============ Single Prediction Form ============ #
with st.form("user_input_form"):
    st.subheader("🔍 Single Customer Prediction")

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

if submit:
    input_dict = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }

    result = predict_campaign_response(input_dict)

    st.subheader("📊 Prediction Result:")
    if result["prediction"] == 1:
        st.success(f"✅ Likely to Subscribe! (Probability: {result['probability_of_yes']})")
    else:
        st.error(f"❌ Not Likely to Subscribe (Probability: {result['probability_of_yes']})")

    # Log single prediction
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
    # os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "single_predictions_log.csv")
    # os.makedirs("logs", exist_ok=True)
    log_data = pd.DataFrame([input_dict | result])
    # log_path = "logs/single_predictions_log.csv"
    if os.path.exists(log_path):
        log_data.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_data.to_csv(log_path, index=False)

# ============ Batch Prediction ============ #
st.markdown("---")
st.subheader("📂 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Run predictions
    predictions = input_df.apply(predict_campaign_response, axis=1)
    pred_df = pd.DataFrame(predictions.tolist())
    result_df = pd.concat([input_df, pred_df], axis=1)

    st.success("✅ Batch prediction completed!")
    st.dataframe(result_df.head(10))

    # Save to logs
    batch_path = os.path.join(LOG_DIR, "batch_predictions.csv")
    result_df.to_csv(batch_path, index=False)
    #os.makedirs("logs", exist_ok=True)
    #result_df.to_csv("logs/batch_predictions.csv", index=False)
    st.info(f"Saved to: `{os.path.relpath(batch_path)}`")


    st.download_button("📥 Download Predictions",
        data=result_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
