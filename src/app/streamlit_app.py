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
# PAGE 3: ANALYSIS DASHBOARD – ENHANCED & HARMONIZED
# ===========================================================================
elif page == "Analysis Dashboard":
    st.markdown("<div class='main-header'>Customer Churn Analysis</div>", unsafe_allow_html=True)
    st.markdown("Interactive exploration of customer behaviors and churn drivers.")

    # --- UNIFIED COLOR PALETTE ---
    CHURN_COLOR = "#FF6B6B"      # Warm Red
    RETAIN_COLOR = "#4ECDC4"     # Teal
    NEUTRAL_BG = "#F7F9FC"       # Light background
    ACCENT = "#FFE66D"           # Yellow accent
    
    COLOR_MAP = {0: RETAIN_COLOR, 1: CHURN_COLOR}
    LABEL_MAP = {0: "Retained", 1: "Churned"}

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Data not found: {DATA_PATH}")
            st.stop()

    df = load_data()

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("### Filter Data")
    
    min_calls, max_calls = int(df['Customer service calls'].min()), int(df['Customer service calls'].max())
    calls_range = st.sidebar.slider("Customer Service Calls", min_calls, max_calls, (0, max_calls))

    intl_options = ["No", "Yes"]
    intl_selection = st.sidebar.multiselect("International Plan", intl_options, default=intl_options)
    intl_mask_vals = [1 if x == "Yes" else 0 for x in intl_selection]

    vmail_selection = st.sidebar.multiselect("Voice Mail Plan", intl_options, default=intl_options)
    vmail_mask_vals = [1 if x == "Yes" else 0 for x in vmail_selection]

    # APPLY FILTERS
    filtered_df = df[
        (df['Customer service calls'].between(*calls_range)) &
        (df['International plan'].isin(intl_mask_vals)) &
        (df['Voice mail plan'].isin(vmail_mask_vals))
    ]

    # --- KEY METRICS WITH STYLED CARDS ---
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>Total Customers</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 32px;'>{len(filtered_df):,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = filtered_df['Churn'].mean()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>Churn Rate</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 32px;'>{churn_rate:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        high_risk_count = (filtered_df['Customer service calls'] > 3).sum()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>High-Risk Callers</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 32px;'>{high_risk_count:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        intl_users = (filtered_df['International plan'] == 1).sum()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>Intl Plan Users</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 32px;'>{intl_users:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- UNIFIED PLOT STYLING ---
    def apply_modern_theme(fig, title):
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, color="white")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F7F9FC',
            font=dict(family="Segoe UI", size=12, color="white"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#DADADA",
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            height=420
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', linecolor='#DADADA')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', linecolor='#DADADA')
        return fig

    # --- ROW 1: USAGE INSIGHTS ---
    st.markdown("### Usage & Service Patterns")
    col_left, col_right = st.columns(2)

    with col_left:
        fig1 = go.Figure()
        
        # Retained customers
        retained = filtered_df[filtered_df['Churn'] == 0]['Total day minutes']
        fig1.add_trace(go.Histogram(
            x=retained,
            name='Retained',
            marker_color=RETAIN_COLOR,
            opacity=0.75,
            nbinsx=30
        ))
        
        # Churned customers
        churned = filtered_df[filtered_df['Churn'] == 1]['Total day minutes']
        fig1.add_trace(go.Histogram(
            x=churned,
            name='Churned',
            marker_color=CHURN_COLOR,
            opacity=0.75,
            nbinsx=30
        ))
        
        fig1.update_layout(barmode='overlay')
        fig1 = apply_modern_theme(fig1, "Usage Distribution: Total Day Minutes")
        fig1.update_xaxes(title_text="Total Day Minutes")
        fig1.update_yaxes(title_text="Number of Customers")
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        # Violin plot for service calls
        box_df = filtered_df.copy()
        box_df['Churn Label'] = box_df['Churn'].map(LABEL_MAP)
        
        fig2 = go.Figure()
        
        for churn_val, label, color in [(0, 'Retained', RETAIN_COLOR), (1, 'Churned', CHURN_COLOR)]:
            data = box_df[box_df['Churn'] == churn_val]['Customer service calls']
            fig2.add_trace(go.Violin(
                y=data,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color
            ))
        
        fig2 = apply_modern_theme(fig2, "Service Calls Distribution by Churn")
        fig2.update_yaxes(title_text="Customer Service Calls")
        st.plotly_chart(fig2, use_container_width=True)

    # --- ROW 2: RISK ANALYSIS ---
    st.markdown("### Risk Segmentation & Plan Analysis")
    col_left_2, col_right_2 = st.columns(2)

    with col_left_2:
        # Enhanced grouped bar chart
        plan_data = filtered_df.groupby(['International plan', 'Voice mail plan'])['Churn'].agg(['mean', 'count']).reset_index()
        plan_data['Plan Combo'] = plan_data.apply(
            lambda x: f"Intl: {'Yes' if x['International plan']==1 else 'No'}, VM: {'Yes' if x['Voice mail plan']==1 else 'No'}", 
            axis=1
        )
        plan_data['Churn Rate %'] = plan_data['mean'] * 100
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=plan_data['Plan Combo'],
            y=plan_data['Churn Rate %'],
            marker=dict(
                color=plan_data['Churn Rate %'],
                colorscale=[[0, RETAIN_COLOR], [0.5, ACCENT], [1, CHURN_COLOR]],
                showscale=True,
                colorbar=dict(title="Churn %")
            ),
            text=[f"{v:.1f}%" for v in plan_data['Churn Rate %']],
            textposition='outside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>',
            customdata=plan_data['count']
        ))
        
        fig3 = apply_modern_theme(fig3, "Churn Rate by Plan Combination")
        fig3.update_xaxes(title_text="Plan Combination", tickangle=-45)
        fig3.update_yaxes(title_text="Churn Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_right_2:
        # Replace treemap with stacked bar chart (more stable)
        risk_df = filtered_df.copy()
        risk_df['Service Tier'] = pd.cut(
            risk_df['Customer service calls'],
            bins=[-1, 1, 3, 10],
            labels=['Low (0-1)', 'Medium (2-3)', 'High (4+)']
        )
        
        # Remove NaN values
        risk_df = risk_df.dropna(subset=['Service Tier'])
        
        if len(risk_df) > 0:
            # Calculate churn rate by service tier
            tier_data = risk_df.groupby(['Service Tier', 'Churn']).size().unstack(fill_value=0)
            tier_data['Total'] = tier_data.sum(axis=1)
            tier_data['Churn_Rate'] = (tier_data[1] / tier_data['Total'] * 100).round(1)
            tier_data = tier_data.reset_index()
            
            fig4 = go.Figure()
            
            # Stacked bars for retained and churned
            fig4.add_trace(go.Bar(
                name='Retained',
                x=tier_data['Service Tier'],
                y=tier_data[0],
                marker_color=RETAIN_COLOR,
                text=tier_data[0],
                textposition='inside'
            ))
            
            fig4.add_trace(go.Bar(
                name='Churned',
                x=tier_data['Service Tier'],
                y=tier_data[1],
                marker_color=CHURN_COLOR,
                text=tier_data[1],
                textposition='inside'
            ))
            
            fig4.update_layout(barmode='stack')
            fig4 = apply_modern_theme(fig4, "Customer Risk by Service Calls")
            fig4.update_xaxes(title_text="Service Call Tier")
            fig4.update_yaxes(title_text="Number of Customers")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No data available for risk segmentation with current filters")

    # --- ROW 3: CORRELATION & INSIGHTS ---
    st.markdown("### Feature Relationships")
    col_left_3, col_right_3 = st.columns([2, 1])

    with col_left_3:
        # Correlation heatmap with better styling
        preferred_cols = [
            'Total day minutes', 'Total eve minutes', 'Total night minutes', 
            'Total intl minutes', 'Customer service calls', 'Account length', 'Churn'
        ]
        existing_cols = [c for c in preferred_cols if c in filtered_df.columns]
        
        corr_matrix = filtered_df[existing_cols].corr().round(2)
        
        fig5 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation", thickness=15, len=0.7)
        ))
        
        fig5 = apply_modern_theme(fig5, "Feature Correlation Matrix")
        fig5.update_xaxes(side="bottom", tickangle=-45)
        fig5.update_layout(height=450)
        st.plotly_chart(fig5, use_container_width=True)

    with col_right_3:
        # Key insights card
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; height: 400px; color: white;'>
            <h3 style='margin-top: 0; font-size: 20px;'>Key Insights</h3>
            <hr style='border-color: rgba(255,255,255,0.3);'>
            <div style='font-size: 14px; line-height: 1.8;'>
                <p><b>Service Calls:</b> Strongest churn predictor</p>
                <p><b>International Plan:</b> High churn risk</p>
                <p><b>Usage Patterns:</b> Extreme users churn more</p>
                <p><b>Account Age:</b> First year is critical</p>
                <p><b>Voicemail:</b> Retention indicator</p>
                <br>
                <p style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin-top: 20px;'>
                    <b>Action:</b> Target customers with 3+ service calls for retention campaigns
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Churn Prediction System © 2025")