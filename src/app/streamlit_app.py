"""
Enhanced Streamlit Application - Churn Prediction System
Professional UI with caching, error handling, and improved visualizations
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
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
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Production-grade MLOps system for customer churn prediction"
    }
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
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Predict Churn", "Monitoring Dashboard", "Analysis Dashboard"]
)

# Customer presets - "Average Customer" removed
CUSTOMER_PRESETS = {
    "High Risk Customer": {
        "Account length": 45,
        "International plan": "Yes",
        "Voice mail plan": "No",
        "Number vmail messages": 0,
        "Total day minutes": 280.0,
        "Total eve minutes": 220.0,
        "Total night minutes": 200.0,
        "Total intl minutes": 15.0,
        "Total day calls": 110,
        "Total eve calls": 100,
        "Total night calls": 95,
        "Total intl calls": 5,
        "Customer service calls": 5
    },
    "Low Risk Customer": {
        "Account length": 180,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 25,
        "Total day minutes": 150.0,
        "Total eve minutes": 130.0,
        "Total night minutes": 140.0,
        "Total intl minutes": 8.0,
        "Total day calls": 90,
        "Total eve calls": 85,
        "Total night calls": 80,
        "Total intl calls": 3,
        "Customer service calls": 1
    }
}

# Cache data loading
@st.cache_data
def load_analysis_data():
    """Load data for analysis dashboard with caching"""
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None

# PAGE 1: CHURN PREDICTION
if page == "Predict Churn":
    st.markdown("<div class='main-header'>Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("Enter customer data to predict churn probability with AI-powered explanations")

    # Preset selector - "Average Customer" removed
    preset = st.selectbox(
        "Load Customer Preset",
        ["Custom", "High Risk Customer", "Low Risk Customer"],
        help="Select a preset to quickly test different customer profiles"
    )
   
    # Get preset values or defaults
    if preset != "Custom":
        preset_values = CUSTOMER_PRESETS[preset]
    else:
        preset_values = {
            "Account length": 100,
            "International plan": "No",
            "Voice mail plan": "No",
            "Number vmail messages": 0,
            "Total day minutes": 180.0,
            "Total eve minutes": 150.0,
            "Total night minutes": 150.0,
            "Total intl minutes": 10.0,
            "Total day calls": 100,
            "Total eve calls": 90,
            "Total night calls": 90,
            "Total intl calls": 3,
            "Customer service calls": 2
        }

    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Usage Metrics")
            total_day_minutes = st.slider(
                "Total day minutes", 0, 400,
                int(preset_values.get("Total day minutes", 180))
            )
            total_eve_minutes = st.slider(
                "Total eve minutes", 0, 400,
                int(preset_values.get("Total eve minutes", 150))
            )
            total_night_minutes = st.slider(
                "Total night minutes", 0, 400,
                int(preset_values.get("Total night minutes", 150))
            )
            total_intl_minutes = st.slider(
                "Total intl minutes", 0, 30,
                int(preset_values.get("Total intl minutes", 10))
            )
           
            total_day_calls = st.slider(
                "Total day calls", 0, 200,
                int(preset_values.get("Total day calls", 100))
            )
            total_eve_calls = st.slider(
                "Total eve calls", 0, 200,
                int(preset_values.get("Total eve calls", 90))
            )
            total_night_calls = st.slider(
                "Total night calls", 0, 200,
                int(preset_values.get("Total night calls", 90))
            )
            total_intl_calls = st.slider(
                "Total intl calls", 0, 30,
                int(preset_values.get("Total intl calls", 3))
            )
        with col2:
            st.subheader("Account Details")
            account_length = st.slider(
                "Account length (days)", 1, 250,
                int(preset_values.get("Account length", 100))
            )
           
            intl_plan_default = preset_values.get("International plan", "No")
            international_plan = st.selectbox(
                "International plan",
                ["No", "Yes"],
                index=1 if intl_plan_default == "Yes" else 0
            )
           
            vmail_plan_default = preset_values.get("Voice mail plan", "No")
            voice_mail_plan = st.selectbox(
                "Voice mail plan",
                ["No", "Yes"],
                index=1 if vmail_plan_default == "Yes" else 0
            )
           
            num_vmail_msgs = st.slider(
                "Number vmail messages", 0, 50,
                int(preset_values.get("Number vmail messages", 0))
            )
           
            customer_service_calls = st.slider(
                "Customer service calls", 0, 9,
                int(preset_values.get("Customer service calls", 1))
            )
           
            # Risk indicators
            if customer_service_calls >= 4:
                st.warning("High service calls detected - elevated churn risk indicator")
            if international_plan == "Yes":
                st.info("International plan customer - monitor closely")
            if customer_service_calls <= 1 and voice_mail_plan == "Yes":
                st.success("Low risk indicators present")

        submitted = st.form_submit_button("Predict Churn", use_container_width=True, type="primary")

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
        with st.spinner("Analyzing customer data..."):
            try:
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to prediction service. Please ensure the API is running:")
                st.code("uvicorn src.api.main:app --reload --port 8000")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API returned an error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
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
                risk_level = "High"
                confidence = "Strong"
            elif prob > 0.5:
                st.warning("MODERATE CHURN RISK")
                risk_level = "Moderate"
                confidence = "Moderate"
            else:
                st.success("LOW CHURN RISK")
                risk_level = "Low"
                confidence = "Strong"
           
            st.metric("Churn Probability", f"{prob:.1%}")
            st.caption(f"Risk Level: {risk_level} | Confidence: {confidence}")

        st.markdown("---")
        st.subheader("AI Model Explanation")
        st.caption("Understanding what drives this prediction")
        shap_df = pd.DataFrame(top_shap)
        fig = go.Figure(go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"],
            orientation='h',
            marker_color=[PEACH_DARK if v > 0 else SAGE_DARK for v in shap_df["shap_value"]],
            text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="Top 6 Feature Impacts on Prediction",
            xaxis_title="SHAP Value (Impact on Churn Probability)",
            yaxis=dict(categoryorder='total ascending'),
            height=450,
            showlegend=False,
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F7F9FC'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Business interpretation
        with st.expander("How to interpret these results"):
            st.markdown("""
            **SHAP values** explain how each feature influences the prediction:
           
            - **Red bars (positive values)**: These factors INCREASE churn probability
            - **Green bars (negative values)**: These factors DECREASE churn probability
            - **Bar length**: Indicates strength of influence
           
            **Action Items:**
            - Focus retention efforts on customers with high positive SHAP values
            - Reinforce behaviors associated with negative SHAP values
            - Monitor the top 3 features for early warning signs
            """)

# PAGE 2: MONITORING DASHBOARD
elif page == "Monitoring Dashboard":
    st.markdown("<div class='main-header'>Model Monitoring Dashboard</div>", unsafe_allow_html=True)
    st.markdown("Real-time performance tracking and drift detection")

    if not MONITORING_DIR.exists() or not list(MONITORING_DIR.glob("monitoring_report_*.json")):
        st.warning("No monitoring data available yet.")
        st.info("Monitoring reports will appear here after running the monitoring pipeline.")
        
        with st.expander("How to generate monitoring data"):
            st.markdown("""
            Run the monitoring pipeline to generate reports:
            
            ```bash
            # Get experiment ID from MLflow
            python -c "import mlflow; client = mlflow.tracking.MlflowClient(); exp = client.get_experiment_by_name('ChurnPrediction-Pipeline'); print(exp.experiment_id)"
            
            # Run monitoring
            python src/pipelines/run_monitoring.py <experiment_id>
            ```
            
            Or run the full pipeline:
            ```bash
            python run_all.py
            ```
            """)
        st.stop()

    try:
        reports = sorted(
            MONITORING_DIR.glob("monitoring_report_*.json"), 
            key=lambda x: x.stat().st_mtime, 
            reverse=True
        )
        
        with open(reports[0]) as f:
            latest_report = json.load(f)
    except Exception as e:
        st.error(f"Error loading monitoring report: {str(e)}")
        st.stop()

    st.caption(f"Last Updated: {latest_report.get('timestamp', 'Unknown')}")

    # Health Status
    if 'performance' in latest_report:
        perf = latest_report['performance']
        roc_auc = perf.get('roc_auc', 0)
        
        if roc_auc >= 0.85:
            health_status = "Healthy"
            health_emoji = ":green_heart:"
        elif roc_auc >= 0.75:
            health_status = "Warning"
            health_emoji = ":yellow_heart:"
        else:
            health_status = "Critical"
            health_emoji = ":red_circle:"
        
        st.markdown(f"**System Health:** {health_emoji} {health_status}")

    st.markdown("<div class='sub-header'>Current Model Performance</div>", unsafe_allow_html=True)
    
    if 'performance' in latest_report:
        perf = latest_report['performance']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1: 
            st.metric("ROC-AUC", f"{perf['roc_auc']:.4f}")
        with col2: 
            st.metric("Accuracy", f"{perf['accuracy']:.4f}")
        with col3: 
            st.metric("Precision", f"{perf['precision']:.4f}")
        with col4: 
            st.metric("Recall", f"{perf['recall']:.4f}")
    else:
        st.info("Performance metrics not available in latest report")

    st.markdown("---")
    st.markdown("<div class='sub-header'>Drift Detection Status</div>", unsafe_allow_html=True)

    feat_drift = latest_report.get('feature_drift', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drift_rate = feat_drift.get('drift_rate', 0) * 100
        if drift_rate > 30:
            st.metric("Feature Drift Rate", f"{drift_rate:.1f}%", delta="Critical", delta_color="inverse")
        elif drift_rate > 10:
            st.metric("Feature Drift Rate", f"{drift_rate:.1f}%", delta="Warning", delta_color="off")
        else:
            st.metric("Feature Drift Rate", f"{drift_rate:.1f}%", delta="Normal", delta_color="normal")
    
    with col2:
        drifted = feat_drift.get('features_drifted', 0)
        total = feat_drift.get('features_checked', 0)
        st.metric("Drifted Features", f"{drifted}/{total}")
    
    with col3:
        pred_drift = latest_report.get('prediction_drift', {}).get('drift_detected', False)
        st.metric("Prediction Drift", "DETECTED" if pred_drift else "NONE")

    # Drift visualization
    if feat_drift.get('drifted_features'):
        st.markdown("**Drifted Features (Top 5):**")
        drifted_list = feat_drift.get('drifted_features', [])[:5]
        for i, feature in enumerate(drifted_list, 1):
            st.text(f"{i}. {feature}")

    st.markdown("---")
    
    # Retraining recommendation
    retrain_info = latest_report.get('retraining', {})
    if retrain_info.get('recommended'):
        st.warning(f"**Retraining Recommended**")
        st.caption(f"Reason: {retrain_info.get('reason', 'Not specified')}")
        
        with st.expander("What does this mean?"):
            st.markdown("""
            The monitoring system has detected conditions that warrant model retraining:
            
            - Performance has degraded below acceptable thresholds
            - Significant data drift has been detected
            - Model predictions are shifting from baseline
            
            **Recommended Action:** Run the retraining pipeline to update the model.
            """)
    else:
        st.success("No retraining needed - model performing within acceptable parameters")

    with st.expander("View Raw Monitoring Report"):
        st.json(latest_report)

# PAGE 3: ANALYSIS DASHBOARD
elif page == "Analysis Dashboard":
    st.markdown("<div class='main-header'>Customer Churn Analysis</div>", unsafe_allow_html=True)
    st.markdown("Interactive exploration of customer behaviors and churn drivers")

    df = load_analysis_data()
    
    if df is None:
        st.error("Analysis data not found. Please ensure the pipeline has been run:")
        st.code("python run_all.py")
        st.stop()

    # Sidebar filters
    st.sidebar.markdown("### Filter Data")
    
    min_calls = int(df['Customer service calls'].min())
    max_calls = int(df['Customer service calls'].max())
    calls_range = st.sidebar.slider(
        "Customer Service Calls", 
        min_calls, max_calls, 
        (min_calls, max_calls)
    )

    intl_options = [0, 1]
    intl_selection = st.sidebar.multiselect(
        "International Plan", 
        intl_options,
        default=intl_options,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    vmail_selection = st.sidebar.multiselect(
        "Voice Mail Plan", 
        intl_options,
        default=intl_options,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # Apply filters
    filtered_df = df[
        (df['Customer service calls'].between(*calls_range)) &
        (df['International plan'].isin(intl_selection)) &
        (df['Voice mail plan'].isin(vmail_selection))
    ]

    # Key metrics
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    
    with col2:
        churn_rate = filtered_df['Churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    with col3:
        high_risk_count = (filtered_df['Customer service calls'] > 3).sum()
        st.metric("High-Risk Callers", f"{high_risk_count:,}")
    
    with col4:
        intl_users = (filtered_df['International plan'] == 1).sum()
        st.metric("Intl Plan Users", f"{intl_users:,}")

    st.markdown("---")

    # Visualizations
    st.markdown("### Usage & Service Patterns")
    col_left, col_right = st.columns(2)

    with col_left:
        # Check if column exists
        if 'Total day minutes' in filtered_df.columns:
            fig1 = go.Figure()
            
            retained = filtered_df[filtered_df['Churn'] == 0]['Total day minutes']
            fig1.add_trace(go.Histogram(
                x=retained,
                name='Retained',
                marker_color=SAGE_DARK,
                opacity=0.75,
                nbinsx=30
            ))
            
            churned = filtered_df[filtered_df['Churn'] == 1]['Total day minutes']
            fig1.add_trace(go.Histogram(
                x=churned,
                name='Churned',
                marker_color=PEACH_DARK,
                opacity=0.75,
                nbinsx=30
            ))
            
            fig1.update_layout(
                barmode='overlay',
                title="Usage Distribution: Total Day Minutes",
                xaxis_title="Total Day Minutes",
                yaxis_title="Number of Customers",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Total day minutes data not available in processed dataset")

    with col_right:
        fig2 = go.Figure()
        
        for churn_val, label, color in [(0, 'Retained', SAGE_DARK), (1, 'Churned', PEACH_DARK)]:
            data = filtered_df[filtered_df['Churn'] == churn_val]['Customer service calls']
            fig2.add_trace(go.Violin(
                y=data,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color
            ))
        
        fig2.update_layout(
            title="Service Calls Distribution by Churn",
            yaxis_title="Customer Service Calls",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Additional insights
    st.markdown("---")
    st.markdown("### Customer Segmentation")
    
    # Create risk segments
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['Risk_Segment'] = pd.cut(
        filtered_df_copy['Customer service calls'],
        bins=[-1, 1, 3, 10],
        labels=['Low Risk (0-1)', 'Medium Risk (2-3)', 'High Risk (4+)']
    )
    
    segment_data = filtered_df_copy.groupby('Risk_Segment')['Churn'].agg(['count', 'mean']).reset_index()
    segment_data.columns = ['Risk Segment', 'Count', 'Churn Rate']
    segment_data['Churn Rate'] = segment_data['Churn Rate'] * 100
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=segment_data['Risk Segment'],
            y=segment_data['Churn Rate'],
            marker_color=[SAGE_DARK, PEACH, PEACH_DARK],
            text=segment_data['Churn Rate'].round(1),
            texttemplate='%{text}%',
            textposition='outside'
        )
    ])
    
    fig3.update_layout(
        title="Churn Rate by Risk Segment",
        xaxis_title="Risk Segment",
        yaxis_title="Churn Rate (%)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)

    # Download option
    st.markdown("---")
    col_download, col_info = st.columns([1, 3])
    with col_download:
        if st.button("Download Filtered Data", type="secondary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"filtered_churn_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    with col_info:
        st.caption(f"Filtered dataset contains {len(filtered_df):,} customers")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Churn Prediction System v1.0")
st.sidebar.caption("Production MLOps Deployment")