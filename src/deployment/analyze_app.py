import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(page_title="Churn Analysis", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"

# Load model, data, features
model = joblib.load(MODEL_DIR / "best_model_final.pkl")
df = pd.read_csv(DATA_PATH)
selected_features = joblib.load(MODEL_DIR / "selected_features.pkl")

# Load evaluation
with open(MODEL_DIR / "evaluation_results.json") as f:
    eval_data = json.load(f)
    best = next(m for m in eval_data["all_results"] if m["model"] == eval_data["best_model"])

st.title("Churn Prediction Analysis Dashboard")
st.markdown("Explore model behavior, feature importance, and prediction trends.")

# 1. MODEL PERFORMANCE
st.header("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{best['accuracy']:.3f}")
    st.metric("Precision", f"{best['precision']:.3f}")
with col2:
    st.metric("Recall", f"{best['recall']:.3f}")
    st.metric("ROC-AUC", f"{best['roc_auc']:.3f}")

# 2. FEATURE IMPORTANCE
st.header("Top Predictive Features")
importances = model.feature_importances_
feat_imp = pd.DataFrame({"Feature": selected_features, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)

fig = px.bar(feat_imp, x="Importance", y="Feature", orientation='h',
             color="Importance", color_continuous_scale="Viridis",
             title="Top 10 Features")
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, use_container_width=True)

# 3. CHURN RATE BY STATE
st.header("Churn Rate by State")
if "State" in df.columns:
    state_churn = df.groupby("State")["Churn"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(x=state_churn.index, y=state_churn.values*100,
                 labels={"x": "State", "y": "Churn Rate (%)"},
                 color=state_churn.values*100, color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("State column not in final dataset (dropped during feature selection).")

# 4. TENURE vs USAGE
st.header("Customer Tenure vs Daily Usage")
required = ["Customer_tenure_months", "Avg_Daily_Usage"]
if all(col in df.columns for col in required):
    fig = px.scatter(df, x="Customer_tenure_months", y="Avg_Daily_Usage", color="Churn",
                     color_discrete_map={0: "lightblue", 1: "red"}, opacity=0.7,
                     title="High Usage + Short Tenure = Risk")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Required features (tenure/usage) not selected in final model.")