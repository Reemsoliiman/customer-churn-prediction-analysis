"""
Streamlit UI → calls FastAPI /predict → shows churn risk + SHAP
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG – change only if you deploy elsewhere
# ------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")  # local default
# When you deploy on Render, set env var API_URL = https://churn-api.onrender.com/predict

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Customer Churn – Predict & Explain")
st.markdown(
    "Enter customer data → get **churn probability** + **SHAP explanation** "
    "powered by a tree model."
)

# ------------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Usage")
        total_day_minutes = st.slider("Total day minutes", 0, 350, 180)
        total_eve_minutes = st.slider("Total eve minutes", 0, 300, 150)
        total_night_minutes = st.slider("Total night minutes", 0, 300, 150)
        total_intl_minutes = st.slider("Total intl minutes", 0, 20, 5)

        total_day_calls = st.slider("Total day calls", 0, 200, 100)
        total_eve_calls = st.slider("Total eve calls", 0, 200, 90)
        total_night_calls = st.slider("Total night calls", 0, 200, 90)
        total_intl_calls = st.slider("Total intl calls", 0, 20, 3)

    with col2:
        st.subheader("Account")
        account_length = st.slider("Account length (days)", 1, 250, 100)
        international_plan = st.selectbox("International plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
        num_vmail_msgs = st.slider("Number vmail messages", 0, 50, 0)
        customer_service_calls = st.slider("Customer service calls", 0, 9, 1)

    submitted = st.form_submit_button("Predict & Explain", use_container_width=True)

# ------------------------------------------------------------------
# CALL FASTAPI
# ------------------------------------------------------------------
def call_api(payload: dict) -> dict | None:
    """POST to FastAPI, return JSON or None on error."""
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        return None

# ------------------------------------------------------------------
# PREDICTION + SHAP DISPLAY
# ------------------------------------------------------------------
if submitted:
    # Build payload **exactly** as FastAPI expects (original column names)
    payload = {
        "Account length": account_length,
        "International plan": international_plan,
        "Voice mail plan": voice_mail_plan,
        "Number vmail messages": num_vmail_msgs,
        "Total day minutes": float(total_day_minutes),
        "Total eve minutes": float(total_eve_minutes),
        "Total night minutes": float(total_night_minutes),
        "Total intl minutes": float(total_intl_minutes),
        "Total day calls": total_day_calls,
        "Total eve calls": total_eve_calls,
        "Total night calls": total_night_calls,
        "Total intl calls": total_intl_calls,
        "Customer service calls": customer_service_calls,
    }

    with st.spinner("Contacting prediction service…"):
        result = call_api(payload)

    if not result:
        st.stop()

    prob = result["churn_probability"]
    pred = result["churn_prediction"]
    top_shap = result["top_shap_features"]

    # ----- Prediction Result -----
    st.markdown("---")
    st.subheader("Prediction")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if pred:
            st.error("**HIGH RISK**")
        else:
            st.success("**LOW RISK**")
        st.metric("Churn Probability", f"{prob:.1%}")

    # ----- SHAP Explanation -----
    st.markdown("---")
    st.subheader("Why the model decided that (SHAP)")

    # Force-plot style bar chart (top-6)
    shap_df = pd.DataFrame(top_shap)
    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation='h',
        marker_color=['#ef4444' if v > 0 else '#3b82f6' for v in shap_df["shap_value"]],
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition='outside'
    ))
    fig.update_layout(
        title="Top 6 feature impacts",
        xaxis_title="SHAP value (impact on churn probability)",
        yaxis=dict(categoryorder='total ascending'),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English summary
    st.markdown("**Plain English**")
    for row in top_shap[:3]:
        impact = "increased" if row["shap_value"] > 0 else "decreased"
        st.markdown(f"• **{row['feature']}** {impact} churn risk by **{abs(row['shap_value']):.3f}**")

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: XGBoost/RandomForest (best from MLflow) | "
    "Feature selection: RFE (20 features) | "
    "Explanations: SHAP TreeExplainer"
)