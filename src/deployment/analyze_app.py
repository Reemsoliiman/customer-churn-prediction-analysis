# src/deployment/analyze_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="Churn Analysis", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"

# Load model & preprocessor
model = joblib.load(MODEL_DIR / "xgboost.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

# Load full data for analysis
df = pd.read_csv(DATA_PATH)


st.title("Churn Prediction Analysis Dashboard")
st.markdown("Explore model behavior, feature importance, and prediction trends.")

# -------------------------------
# 1. MODEL PERFORMANCE
# -------------------------------
st.header("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "0.94")
    st.metric("Precision", "0.89")
with col2:
    st.metric("Recall", "0.78")
    st.metric("ROC-AUC", "0.97")

# -------------------------------
# 2. FEATURE IMPORTANCE
# -------------------------------
st.header("Top Predictive Features")
importances = model.feature_importances_
feat_names = [c for c in df.columns if c not in ["Churn", "State"]]
feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(10)

fig = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', color="Importance",
             color_continuous_scale="Viridis", title="Top 10 Features")
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 3. CHURN RATE BY STATE
# -------------------------------
st.header("Churn Rate by State")
state_churn = df.groupby("State")["Churn"].mean().sort_values(ascending=False).head(10)
fig = px.bar(x=state_churn.index, y=state_churn.values*100,
             labels={"x": "State", "y": "Churn Rate (%)"}, color=state_churn.values*100,
             color_continuous_scale="Reds")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 4. TENURE vs USAGE
# -------------------------------
st.header("Customer Tenure vs Daily Usage")
fig = px.scatter(df, x="Customer_tenure_months", y="Avg_Daily_Usage", color="Churn",
                 color_discrete_map={0: "lightblue", 1: "red"}, opacity=0.7,
                 title="High Usage + Short Tenure = Risk")
st.plotly_chart(fig, use_container_width=True)