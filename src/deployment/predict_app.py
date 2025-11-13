# src/deployment/predict_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Predict Churn", layout="centered")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

model = joblib.load(MODEL_DIR / "xgboost.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

# Load feature names
df_sample = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv")
feature_names = [c for c in df_sample.columns if c not in ["Churn", "State"]]

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values

# -------------------------------
# TITLE
# -------------------------------
st.title("Customer Churn Prediction")
st.markdown("Enter customer details to predict churn risk.")

# -------------------------------
# INPUT FORM
# -------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_day_minutes = st.slider("Total Day Minutes", 0, 350, 180)
        total_eve_minutes = st.slider("Total Eve Minutes", 0, 300, 150)
        total_night_minutes = st.slider("Total Night Minutes", 0, 300, 150)
        total_intl_minutes = st.slider("Total Intl Minutes", 0, 20, 5)
        customer_service_calls = st.slider("Customer Service Calls", 0, 9, 1)

    with col2:
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        account_length = st.slider("Account Length (days)", 1, 250, 100)
        num_vmail_msgs = st.slider("Voice Mail Messages", 0, 50, 0)

    submitted = st.form_submit_button("Predict Churn Risk")

# -------------------------------
# PREDICTION
# -------------------------------
if submitted:
    # Build input
    input_dict = {
        "Account length": account_length,
        "International plan_Yes": 1 if international_plan == "Yes" else 0,
        "Voice mail plan_Yes": 1 if voice_mail_plan == "Yes" else 0,
        "Number vmail messages": num_vmail_msgs,
        "Total day minutes": total_day_minutes,
        "Total eve minutes": total_eve_minutes,
        "Total night minutes": total_night_minutes,
        "Total intl minutes": total_intl_minutes,
        "Customer service calls": customer_service_calls,
        "High_Service_Calls": 1 if customer_service_calls > 3 else 0
    }

    # Engineer features
    input_dict["Customer_tenure_months"] = account_length / 30.0
    input_dict["Total_Minutes"] = (total_day_minutes + total_eve_minutes +
                                   total_night_minutes + total_intl_minutes)
    input_dict["Avg_Daily_Usage"] = input_dict["Total_Minutes"] / (input_dict["Customer_tenure_months"] + 1)

    input_df = pd.DataFrame([input_dict])[feature_names]
    X_scaled = preprocessor.transform(input_df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob > 0.5)

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"**HIGH RISK** — Customer is likely to churn (Probability: {prob:.1%})")
    else:
        st.success(f"**LOW RISK** — Customer is likely to stay (Probability: {prob:.1%})")

    # -------------------------------
    # SHAP EXPLANATION
    # -------------------------------
    st.subheader("Why This Prediction?")
    shap_val = explainer.shap_values(X_scaled)[0]
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_val
    }).sort_values("SHAP Value", key=abs, ascending=False).head(6)

    fig = go.Figure(go.Bar(
        x=shap_df["SHAP Value"],
        y=shap_df["Feature"],
        orientation='h',
        marker_color=np.where(shap_df["SHAP Value"] > 0, 'red', 'blue')
    ))
    fig.update_layout(title="Top Factors Driving Prediction", yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)