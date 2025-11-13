import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("models/trained_models/xgboost_selected_features.pkl")
preprocessor = joblib.load("models/trained_models/preprocessor_selected.pkl")

selected_features = [
    'State_VA', 'Total day charge', 'State_AZ', 'State_MD',
    'High_Customer_Service', 'State_RI', 'Total day minutes',
    'Total_Charge', 'State_IL', 'Total_Minutes', 'State_MT',
    'Customer service calls', 'International plan_Yes', 'State_NJ',
    'State_TX', 'State_HI', 'Voice mail plan_Yes'
]

st.title("📉 Customer Churn Prediction App")
st.write("Please input the customer's data to predict whether they will churn (leave the company) or remain.")

col1, col2 = st.columns(2)

with col1:
    total_day_charge = st.number_input("Total Day Charge", min_value=0.0, value=30.0)
    total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=200.0)
    total_charge = st.number_input("Total Charge", min_value=0.0, value=70.0)
    total_minutes = st.number_input("Total Minutes", min_value=0.0, value=300.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)

with col2:
    state = st.selectbox("State", ['VA', 'AZ', 'MD', 'RI', 'IL', 'MT', 'NJ', 'TX', 'HI'])
    international_plan = st.selectbox("International Plan", ['No', 'Yes'])
    voice_mail_plan = st.selectbox("Voice Mail Plan", ['No', 'Yes'])

high_service = 1 if customer_service_calls > 3 else 0

input_data = {
    'Total day charge': total_day_charge,
    'Total day minutes': total_day_minutes,
    'Total_Charge': total_charge,
    'Total_Minutes': total_minutes,
    'Customer service calls': customer_service_calls,
    'High_Customer_Service': high_service,
    'International plan_Yes': 1 if international_plan == 'Yes' else 0,
    'Voice mail plan_Yes': 1 if voice_mail_plan == 'Yes' else 0,
}

for s in ['VA', 'AZ', 'MD', 'RI', 'IL', 'MT', 'NJ', 'TX', 'HI']:
    input_data[f'State_{s}'] = 1 if state == s else 0

input_df = pd.DataFrame([input_data])[selected_features]

if st.button("🔍 Predict"):
    try:
        X_scaled = preprocessor.transform(input_df)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 The customer is likely to churn [leave the company]. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ The customer is likely to remain with the company. (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction {e}")
