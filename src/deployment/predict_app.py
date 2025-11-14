# src/deployment/predict_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.graph_objects as go
from pathlib import Path

# Suppress sklearn warning
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Predict Churn", layout="centered")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

# Load model and selected features
model = joblib.load(MODEL_DIR / "best_model_final.pkl")
selected_features = joblib.load(MODEL_DIR / "selected_features.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# -------------------------------
# TITLE
# -------------------------------
st.title("Customer Churn Prediction")
st.markdown("Enter customer details to predict churn risk.")

# -------------------------------
# FEATURE ENGINEERING FUNCTION
# -------------------------------
def create_engineered_input(data):
    df = pd.DataFrame([data])
    
    # Replicate feature_engineering.py logic
    df["Customer_tenure_months"] = df["Account length"] / 30.0
    df["Total_Minutes"] = (
        df["Total day minutes"] + df["Total eve minutes"] +
        df["Total night minutes"] + df["Total intl minutes"]
    )
    df["Total_Calls"] = (
        df["Total day calls"] + df["Total eve calls"] +
        df["Total night calls"] + df["Total intl calls"]
    )
    df["Avg_Daily_Usage"] = df["Total_Minutes"] / (df["Customer_tenure_months"] + 1)
    df["Call_Frequency"] = df["Total_Calls"] / (df["Customer_tenure_months"] + 1)
    df["Intl_Usage_Rate"] = df["Total intl minutes"] / (df["Total_Minutes"] + 1)
    df["High_Service_Calls"] = (df["Customer service calls"] > 3).astype(int)
    df["Has_Vmail"] = (df["Number vmail messages"] > 0).astype(int)

    # Log features (safe)
    log_cols = ["Total_Minutes", "Total_Calls", "Avg_Daily_Usage", "Call_Frequency"]
    df[log_cols] = df[log_cols].clip(lower=0).fillna(0)
    for col in log_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    # Drop charge columns (if present)
    charge_cols = [c for c in df.columns if "charge" in c.lower()]
    df = df.drop(columns=charge_cols, errors="ignore")

    return df

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
        total_day_calls = st.slider("Total Day Calls", 0, 200, 100)
        total_eve_calls = st.slider("Total Eve Calls", 0, 200, 90)
        total_night_calls = st.slider("Total Night Calls", 0, 200, 90)
        total_intl_calls = st.slider("Total Intl Calls", 0, 20, 3)

    with col2:
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        account_length = st.slider("Account Length (days)", 1, 250, 100)
        num_vmail_msgs = st.slider("Voice Mail Messages", 0, 50, 0)

    submitted = st.form_submit_button("Predict Churn Risk")

# -------------------------------
# PREDICTION + SHAP
# -------------------------------
if submitted:
    # Build raw input
    raw_input = {
        "Account length": account_length,
        "International plan_Yes": 1 if international_plan == "Yes" else 0,
        "Voice mail plan_Yes": 1 if voice_mail_plan == "Yes" else 0,
        "Number vmail messages": num_vmail_msgs,
        "Total day minutes": total_day_minutes,
        "Total eve minutes": total_eve_minutes,
        "Total night minutes": total_night_minutes,
        "Total intl minutes": total_intl_minutes,
        "Customer service calls": customer_service_calls,
        "Total day calls": total_day_calls,
        "Total eve calls": total_eve_calls,
        "Total night calls": total_night_calls,
        "Total intl calls": total_intl_calls,
    }

    # Engineer features
    engineered_df = create_engineered_input(raw_input)
    input_aligned = engineered_df.reindex(columns=selected_features, fill_value=0)
    X_input = input_aligned.values

    # Predict
    prob = model.predict_proba(X_input)[0][1]
    pred = int(prob > 0.5)

    # RESULT
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"**HIGH RISK** — Customer is likely to churn (Probability: {prob:.1%})")
    else:
        st.success(f"**LOW RISK** — Customer is likely to stay (Probability: {prob:.1%})")

    # SHAP EXPLANATION
    st.subheader("Why This Prediction?")
    try:
        shap_vals = explainer.shap_values(X_input)

        # Normalize SHAP output for positive class (churn)
        if isinstance(shap_vals, list):
            shap_val = shap_vals[1].flatten() if len(shap_vals) > 1 else shap_vals[0].flatten()
        else:
            shap_val = shap_vals.flatten()

        # Validate length
        if len(shap_val) != len(selected_features):
            st.warning(f"SHAP length mismatch: {len(shap_val)} vs {len(selected_features)}")
        else:
            shap_df = pd.DataFrame({
                "Feature": selected_features,
                "SHAP Value": shap_val
            }).sort_values("SHAP Value", key=abs, ascending=False).head(6)

            fig = go.Figure(go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Feature"],
                orientation='h',
                marker_color=np.where(shap_df["SHAP Value"] > 0, 'red', 'blue')
            ))
            fig.update_layout(
                title="Top Factors Driving Prediction",
                yaxis={'categoryorder': 'total ascending'},
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"SHAP explanation failed: {str(e)}")
        st.info("This may occur due to model compatibility. Try different inputs.")