"""
Streamlit app for churn prediction with SHAP explanations.
Uses shared helper functions to ensure consistency with training.
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.helpers import engineer_features, align_features_for_prediction

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Churn Predict & Explain", layout="wide")

ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

@st.cache_resource
def load_model_and_features():
    model = joblib.load(ARTIFACTS_DIR / "best_model_final.pkl")
    selected_features = joblib.load(ARTIFACTS_DIR / "selected_features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, selected_features, explainer

model, selected_features, explainer = load_model_and_features()

# -------------------------------------------------
# UI – INPUT FORM
# -------------------------------------------------
st.title("Customer Churn – Predict & Explain")
st.markdown("Enter customer data → get **churn risk** + **explanation** of the model's decision.")

with st.form("main_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Usage Information")
        total_day_minutes = st.slider("Total Day Minutes", 0, 350, 180)
        total_eve_minutes = st.slider("Total Eve Minutes", 0, 300, 150)
        total_night_minutes = st.slider("Total Night Minutes", 0, 300, 150)
        total_intl_minutes = st.slider("Total Intl Minutes", 0, 20, 5)

        total_day_calls = st.slider("Total Day Calls", 0, 200, 100)
        total_eve_calls = st.slider("Total Eve Calls", 0, 200, 90)
        total_night_calls = st.slider("Total Night Calls", 0, 200, 90)
        total_intl_calls = st.slider("Total Intl Calls", 0, 20, 3)

    with col2:
        st.subheader("Customer Information")
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        account_length = st.slider("Account Length (days)", 1, 250, 100)
        num_vmail_msgs = st.slider("Voice Mail Messages", 0, 50, 0)
        customer_service_calls = st.slider("Customer Service Calls", 0, 9, 1)

    submitted = st.form_submit_button("Predict & Explain", use_container_width=True)

# -------------------------------------------------
# PREDICTION + SHAP
# -------------------------------------------------
if submitted:
    raw_input = {
        "Account length": account_length,
        "International plan": 1 if international_plan == "Yes" else 0,
        "Voice mail plan": 1 if voice_mail_plan == "Yes" else 0,
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

    df_input = pd.DataFrame([raw_input])
    df_engineered = engineer_features(df_input, is_training=False)
    X_aligned = align_features_for_prediction(df_engineered, selected_features)
    X_np = X_aligned.values

    prob = model.predict_proba(X_np)[0, 1]
    pred = int(prob > 0.5)

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if pred:
            st.error("### HIGH RISK")
            st.metric("Churn Probability", f"{prob:.1%}")
        else:
            st.success("### LOW RISK")
            st.metric("Churn Probability", f"{prob:.1%}")

    # ---------- SHAP ----------
    try:
        st.markdown("---")
        st.subheader("Model Explanation (SHAP)")

        shap_vals = explainer.shap_values(X_np)

        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_class1 = shap_vals[1].flatten()
            base_value = explainer.expected_value[1]
        else:
            shap_class1 = shap_vals.flatten()
            base_value = explainer.expected_value

        if len(shap_class1) != len(selected_features):
            st.error(
                f"SHAP length mismatch: {len(shap_class1)} vs {len(selected_features)}. "
                "Please retrain the model."
            )
        else:
            # Force plot
            st.markdown("#### SHAP Force Plot")
            st.caption("Shows how each feature pushed the prediction higher or lower.")
            force_html = shap.force_plot(
                base_value,
                shap_class1,
                X_np,
                feature_names=selected_features,
                matplotlib=False,
                show=False,
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{force_html.html()}</body>"
            st.components.v1.html(shap_html, height=200, scrolling=True)

            # Top-6 bar chart
            st.markdown("#### Top 6 Feature Impacts")
            shap_df = pd.DataFrame({
                "Feature": selected_features,
                "SHAP": shap_class1
            }).assign(Abs=lambda d: d["SHAP"].abs())\
              .sort_values("Abs", ascending=False).head(6)

            fig = go.Figure(go.Bar(
                x=shap_df["SHAP"],
                y=shap_df["Feature"],
                orientation='h',
                marker_color=['#ef4444' if v > 0 else '#3b82f6' for v in shap_df["SHAP"]],
                text=[f"{v:+.3f}" for v in shap_df["SHAP"]],
                textposition='outside'
            ))
            fig.update_layout(
                title="Features that increased (+) or decreased (-) churn risk",
                xaxis_title="SHAP Value",
                yaxis=dict(categoryorder='total ascending'),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Plain-English
            st.markdown("#### Plain English")
            top3 = shap_df.head(3)
            for _, row in top3.iterrows():
                impact = "increased" if row['SHAP'] > 0 else "decreased"
                st.markdown(
                    f"**{row['Feature']}** {impact} churn risk by **{abs(row['SHAP']):.3f}**"
                )

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        st.info("Make sure the saved model is a tree-based classifier (RandomForest / XGBoost).")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Model trained on Telco Churn dataset | "
    "Feature selection: RFE (20 features) | "
    "Explanations: SHAP TreeExplainer"
)