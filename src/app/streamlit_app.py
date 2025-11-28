"""
Complete Streamlit Application - Churn Prediction System
Pages: Predict Churn | Monitoring Dashboard | Analysis Dashboard
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONITORING_DIR = PROJECT_ROOT / "monitoring"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"

# Colors
PEACH_DARK = '#FF9A76'
SAGE_DARK = '#7A9B57'
SAGE = '#A8C686'
PEACH = '#FFCBA4'
NEUTRAL = '#F5F5DC'

# Page config
st.set_page_config(
    page_title="Churn Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Predict Churn", "Monitoring Dashboard", "Analysis Dashboard"]
)

# ===========================================================================
# PAGE 1: CHURN PREDICTION
# ===========================================================================
if page == "Predict Churn":
    st.markdown("<div class='main-header'>Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("Enter customer data to predict churn probability with SHAP explanations")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Usage Metrics")
            total_day_minutes = st.slider("Total day minutes", 0, 400, 180)
            total_eve_minutes = st.slider("Total eve minutes", 0, 400, 150)
            total_night_minutes = st.slider("Total night minutes", 0, 400, 150)
            total_intl_minutes = st.slider("Total intl minutes", 0, 30, 10)
            total_day_calls = st.slider("Total day calls", 0, 200, 100)
            total_eve_calls = st.slider("Total eve calls", 0, 200, 90)
            total_night_calls = st.slider("Total night calls", 0, 200, 90)
            total_intl_calls = st.slider("Total intl calls", 0, 30, 3)

        with col2:
            st.subheader("Account Details")
            account_length = st.slider("Account length (days)", 1, 250, 100)
            international_plan = st.selectbox("International plan", ["No", "Yes"])
            voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])
            num_vmail_msgs = st.slider("Number vmail messages", 0, 50, 0)
            customer_service_calls = st.slider("Customer service calls", 0, 9, 1)

        submitted = st.form_submit_button("Predict and Explain", use_container_width=True)

    if submitted:
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

        with st.spinner("Contacting prediction service..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Start with:\n`uvicorn src.api.main:app --reload --port 8000`")
                st.stop()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.info("Make sure the API is running and the model is trained.")
                st.stop()

        prob = result["churn_probability"]
        pred = result["churn_prediction"]
        top_shap = result["top_shap_features"]

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if pred:
                st.error("HIGH CHURN RISK")
            else:
                st.success("LOW CHURN RISK")
            st.metric("Churn Probability", f"{prob:.1%}")

        st.markdown("---")
        st.subheader("Model Explanation (SHAP Values)")

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
            title="Top 6 Feature Impacts on Churn Prediction",
            xaxis_title="SHAP Value (Impact on Probability)",
            yaxis=dict(categoryorder='total ascending'),
            height=400,
            showlegend=False,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # st.markdown("**Plain English Explanation:**")
        # for i, row in enumerate(top_shap[:3], 1):
        #     impact = "increased" if row["shap_value"] > 0 else "decreased"
        #     st.markdown(f"{i}. **{row['feature']}** {impact} churn risk by **{abs(row['shap_value']):.3f}**")

# ===========================================================================
# PAGE 2: MONITORING DASHBOARD
# ===========================================================================
elif page == "Monitoring Dashboard":
    st.markdown("<div class='main-header'>Model Monitoring Dashboard</div>", unsafe_allow_html=True)
    st.markdown("Real-time performance tracking and drift detection")

    if not MONITORING_DIR.exists() or not list(MONITORING_DIR.glob("monitoring_report_*.json")):
        st.warning("No monitoring data available yet.")
        st.info("Run the monitoring pipeline first:")
        st.code("python src/pipelines/run_monitoring.py <experiment_id>")
        st.info("Or run the full pipeline:")
        st.code("python run_all.py")
        st.stop()

    reports = sorted(MONITORING_DIR.glob("monitoring_report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    with open(reports[0]) as f:
        latest_report = json.load(f)

    st.caption(f"Last Updated: {latest_report.get('timestamp', 'Unknown')}")

    st.markdown("<div class='sub-header'>Current Model Performance</div>", unsafe_allow_html=True)
    if 'performance' in latest_report:
        perf = latest_report['performance']
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("ROC-AUC", f"{perf['roc_auc']:.4f}")
        with col2: st.metric("Accuracy", f"{perf['accuracy']:.4f}")
        with col3: st.metric("Precision", f"{perf['precision']:.4f}")
        with col4: st.metric("Recall", f"{perf['recall']:.4f}")
    else:
        st.info("Performance metrics not available in latest report")

    st.markdown("---")
    st.markdown("<div class='sub-header'>Drift Detection Status</div>", unsafe_allow_html=True)

    feat_drift = latest_report.get('feature_drift', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        drift_rate = feat_drift.get('drift_rate', 0) * 100
        status = "CRITICAL" if drift_rate > 30 else "WARNING" if drift_rate > 10 else "NORMAL"
        st.metric("Feature Drift Rate", f"{drift_rate:.1f}%", help=f"Status: {status}")
    with col2:
        drifted = feat_drift.get('features_drifted', 0)
        total = feat_drift.get('features_checked', 0)
        st.metric("Drifted Features", f"{drifted}/{total}")
    with col3:
        pred_drift = latest_report.get('prediction_drift', {}).get('drift_detected', False)
        st.metric("Prediction Drift", "DETECTED" if pred_drift else "NONE")

    # Add more monitoring visualizations here if desired (trends, drifted features, etc.)

    st.markdown("---")
    retrain_info = latest_report.get('retraining', {})
    if retrain_info.get('recommended'):
        st.warning(f"Retraining Recommended: {retrain_info.get('reason', 'Not specified')}")
    else:
        st.success("No retraining needed – model performing well")

    with st.expander("View Raw Monitoring Report"):
        st.json(latest_report)

# ===========================================================================
# PAGE 3: ANALYSIS DASHBOARD – FINAL & FIXED
# ===========================================================================
elif page == "Analysis Dashboard":
    st.markdown("<div class='main-header'>Customer Churn Analysis</div>", unsafe_allow_html=True)
    st.markdown("Interactive exploration of customer behaviors and churn drivers.")

    # --- 1. CONFIGURATION & STYLE ---
    # RESTORED: Your preferred Green/Red palette
    CHURN_COLOR = "#ef4444"  # Red
    RETAIN_COLOR = "#10b981" # Green
    COLOR_MAP = {0: RETAIN_COLOR, 1: CHURN_COLOR}
    LABEL_MAP = {0: "No (Retained)", 1: "Yes (Churned)"}

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Data not found: {DATA_PATH}")
            st.stop()

    df = load_data()

    # --- 2. SIDEBAR FILTERS ---
    st.sidebar.markdown("### 🔍 Filter Data")
    
    # Filter 1: Customer Service Calls
    min_calls, max_calls = int(df['Customer service calls'].min()), int(df['Customer service calls'].max())
    calls_range = st.sidebar.slider("Customer Service Calls", min_calls, max_calls, (0, max_calls))

    # Filter 2: International Plan
    intl_options = ["No", "Yes"]
    intl_selection = st.sidebar.multiselect("International Plan", intl_options, default=intl_options)
    intl_mask_vals = [1 if x == "Yes" else 0 for x in intl_selection]

    # Filter 3: Voice Mail Plan
    vmail_selection = st.sidebar.multiselect("Voice Mail Plan", intl_options, default=intl_options)
    vmail_mask_vals = [1 if x == "Yes" else 0 for x in vmail_selection]

    # APPLY FILTERS (On Numeric Data)
    filtered_df = df[
        (df['Customer service calls'].between(*calls_range)) &
        (df['International plan'].isin(intl_mask_vals)) &
        (df['Voice mail plan'].isin(vmail_mask_vals))
    ]

    # --- 3. KEY METRICS ROW ---
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    
    with col2:
        churn_rate = filtered_df['Churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}", delta_color="inverse")

    with col3:
        high_risk_count = (filtered_df['Customer service calls'] > 3).sum()
        st.metric("High-Risk Callers (>3)", f"{high_risk_count:,}")

    with col4:
        intl_users = (filtered_df['International plan'] == 1).sum()
        st.metric("Intl Plan Users", f"{intl_users:,}")

    st.markdown("---")

    # --- 4. VISUALIZATIONS ---
    def clean_layout(fig, title):
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, sans-serif"),
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_xaxes(showgrid=False, linecolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        return fig

    col_left, col_right = st.columns(2)

    # PLOT 1: Distribution of Day Minutes
    with col_left:
        fig1 = px.histogram(
            filtered_df, 
            x="Total day minutes", 
            color="Churn",
            nbins=40,
            opacity=0.85,
            color_discrete_map=COLOR_MAP
        )
        new_names = {"0": "Retained", "1": "Churned"}
        fig1.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
        fig1 = clean_layout(fig1, "Usage Distribution: Total Day Minutes")
        st.plotly_chart(fig1, use_container_width=True)

    # PLOT 2: Customer Service Calls
    with col_right:
        box_df = filtered_df.copy()
        box_df['Churn Label'] = box_df['Churn'].map(LABEL_MAP)
        
        fig2 = px.box(
            box_df, 
            x="Churn Label", 
            y="Customer service calls", 
            color="Churn Label",
            color_discrete_map={"No (Retained)": RETAIN_COLOR, "Yes (Churned)": CHURN_COLOR}
        )
        fig2 = clean_layout(fig2, "Service Calls Impact on Churn")
        st.plotly_chart(fig2, use_container_width=True)

    col_left_2, col_right_2 = st.columns(2)

    # PLOT 3: Sunburst Chart
    with col_left_2:
        sun_data = filtered_df.groupby(["International plan", "Voice mail plan"])["Churn"].agg(['mean', 'count']).reset_index()
        sun_data['Churn Rate'] = sun_data['mean']
        sun_data['International Plan'] = sun_data['International plan'].map({0: "No Intl", 1: "Intl Plan"})
        sun_data['Voice Mail'] = sun_data['Voice mail plan'].map({0: "No VM", 1: "VM Plan"})
        
        fig3 = px.sunburst(
            sun_data, 
            path=["International Plan", "Voice Mail"], 
            values='count',
            color="Churn Rate",
            color_continuous_scale="RdYlBu_r", 
            hover_data=["Churn Rate"]
        )
        fig3.update_layout(title="Churn Risk by Plan Combination", font=dict(family="Segoe UI"))
        st.plotly_chart(fig3, use_container_width=True)

    # PLOT 4: Correlation Heatmap (FIXED CRASH)
    with col_right_2:
        # Define preferred columns
        preferred_cols = [
            'Total day minutes', 'Total eve minutes', 'Total night minutes', 
            'Total intl minutes', 'Customer service calls', 'Account length', 'Churn'
        ]
        # FIX: Only use columns that ACTUALLY exist in your data
        existing_cols = [c for c in preferred_cols if c in filtered_df.columns]
        
        corr_matrix = filtered_df[existing_cols].corr().round(2)
        
        fig4 = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1
        )
        fig4.update_layout(title="Feature Correlation Heatmap", font=dict(family="Segoe UI"))
        st.plotly_chart(fig4, use_container_width=True)
    
# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Churn Prediction System © 2025")