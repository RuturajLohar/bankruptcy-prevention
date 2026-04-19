import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(
    page_title="Bankruptcy Prevention AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    if os.path.exists('bankruptcy_final.csv'):
        return pd.read_csv('bankruptcy_final.csv')
    return None

@st.cache_resource
def load_model():
    if os.path.exists('bankruptcy_model.joblib'):
        return joblib.load('bankruptcy_model.joblib')
    return None

df = load_data()
model = load_model()

# --- Title Section ---
st.title("🏦 Bankruptcy Prevention AI Dashboard")
st.markdown("### Analyzing Business Risk Factors with Machine Learning")
st.divider()

# --- Sidebar: Interactive Prediction ---
st.sidebar.header("🕹️ Risk Calculator")
st.sidebar.markdown("Adjust the risk factors to see the bankruptcy probability.")

with st.sidebar:
    ind_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, 0.5)
    mgt_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, 0.5)
    fin_flex = st.slider("Financial Flexibility", 0.0, 1.0, 1.0, 0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5, 0.5)
    comp = st.slider("Competitiveness", 0.0, 1.0, 1.0, 0.5)
    op_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, 0.5)

    input_data = pd.DataFrame([{
        "industrial_risk": ind_risk,
        "management_risk": mgt_risk,
        "financial_flexibility": fin_flex,
        "credibility": credibility,
        "competitiveness": comp,
        "operating_risk": op_risk
    }])

    if model:
        prob = model.predict_proba(input_data)[0]
        bankruptcy_prob = prob[0]
        healthy_prob = prob[1]
        
        # Gauge Chart for Sidebar
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bankruptcy_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Bankruptcy Risk (%)", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#e74c3c" if bankruptcy_prob > 0.5 else "#27ae60"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4efdf"},
                    {'range': [30, 70], 'color': "#fcf3cf"},
                    {'range': [70, 100], 'color': "#fadbd8"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        status = "🔴 Bankruptcy Warning" if bankruptcy_prob > 0.5 else "🟢 Low Risk / Healthy"
        st.markdown(f"#### Status: {status}")
    else:
        st.error("Model not found. Please train the model first.")

# --- Main Dashboard Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Model Performance Comparison")
    # Accuracy data from recent training logs
    performance_data = {
        'Model': ['KNN', 'Random Forest', 'SVM', 'Logistic Regression', 'Gradient Boosting', 'Neural Network', 'Decision Tree', 'AdaBoost', 'Naive Bayes', 'Perceptron'],
        'Accuracy (%)': [99.09, 99.09, 99.00, 99.00, 98.18, 98.00, 97.27, 96.36, 96.36, 94.55],
        'Consistency': ['High', 'High', 'High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Low', 'Low']
    }
    perf_df = pd.DataFrame(performance_data)
    
    fig_perf = px.bar(
        perf_df, 
        x='Model', 
        y='Accuracy (%)', 
        color='Accuracy (%)',
        color_continuous_scale='Viridis',
        text='Accuracy (%)',
        title="10-Fold Cross-Validation Accuracy"
    )
    fig_perf.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_perf.update_layout(yaxis_range=[90, 100], height=400)
    st.plotly_chart(fig_perf, use_container_width=True)

with col2:
    st.subheader("🎯 Key Risk Factors")
    if model and hasattr(model, 'feature_importances_'):
        # Only works if RF was chosen as best
        importance = model.feature_importances_
        feat_df = pd.DataFrame({'Factor': input_data.columns, 'Importance': importance})
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        fig_feat = px.bar(feat_df, y='Factor', x='Importance', orientation='h', title="Feature Importance")
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        # Fallback explanation if KNN (no feature importance) is best
        st.info("KNN is the current champion. It evaluates risk based on similarity to historical cases.")
        st.markdown("""
        **Top Factors identified by analysis:**
        1. **Competitiveness** (The #1 Predictor)
        2. **Financial Flexibility**
        3. **Credibility**
        """)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("🔗 Risk Correlations")
    if df is not None:
        corr = df.corr()
        fig_corr = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap (Risk Linkages)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Dataset not found for correlation analysis.")

with col4:
    st.subheader("📉 Data Distribution")
    if df is not None:
        dist_df = df['class'].value_counts().reset_index()
        dist_df.columns = ['Status', 'Count']
        dist_df['Status'] = dist_df['Status'].map({0: 'Bankruptcy', 1: 'Non-Bankruptcy'})
        fig_pie = px.pie(dist_df, values='Count', names='Status', hole=.4, color_discrete_sequence=['#e74c3c', '#27ae60'])
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Dataset not found for distribution analysis.")

st.markdown("---")
st.caption("AI Bankruptcy Prevention System • Prepared by Ruturaj • Version 1.2")
