import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. SETUP ---
st.set_page_config(
    page_title="LoanCheck XAI | Credit Risk Intelligence", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL CUSTOM CSS ---
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    
    /* Card styling for all containers */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #1a2c3e 0%, #0f1c2a 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,123,255,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,123,255,0.4);
        background: linear-gradient(135deg, #0056b3 0%, #004094 100%);
    }
    
    /* Input field styling */
    .stNumberInput input, .stSlider, .stSelectbox select {
        border-radius: 10px;
        border: 1px solid #e0e4e8;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.1);
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a2c3e;
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #6c757d;
    }
    
    /* Success/Error cards */
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        font-weight: 600;
        color: #1a2c3e;
        border: 1px solid #e0e4e8;
    }
    
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        border-radius: 0 0 10px 10px;
        padding: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d4e8ff 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    
    /* Footer */
    .footer {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0;
        background: linear-gradient(90deg, transparent, #007bff, transparent);
        height: 2px;
        border: none;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    /* Custom container borders */
    .css-1r6slb0 {
        border-radius: 15px;
        background: white;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_assets():
    model = joblib.load('gb_loan_model.pkl')
    explainer = shap.Explainer(model)
    return model, explainer

model, explainer = load_assets()

# --- 3. HEADER SECTION WITH MODERN DESIGN ---
st.markdown("""
    <div style="background: linear-gradient(135deg, #1a2c3e 0%, #0f1c2a 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="background: #007bff; width: 60px; height: 60px; border-radius: 30px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 30px;">💰</span>
                </div>
                <div>
                    <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">LoanCheck XAI</h1>
                    <p style="color: #a0c4ff; margin: 0; font-size: 1rem;">Transparent Credit Risk Intelligence System</p>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 50px;">
                <span style="color: white;">🎯 98.36% Accuracy</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Developer info banner
st.markdown("""
    <div class="info-box">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">💡</span>
            <div>
                <strong>Developed by: Zohaib Ahmed | MS (Data Science) | Data Scientist</strong><br>
                <small>Powered by G-B with SHAP explanations for full transparency</small>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 4. MAIN INPUT SECTION (Enhanced Layout) ---
st.markdown("## 📋 Applicant Details")
st.markdown("*Please enter the complete financial and personal details below for accurate assessment*")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["👤 Personal Information", "💰 Financial Details", "🏦 Asset Breakdown"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 👨‍👩‍👧 Dependents")
        dependents = st.number_input("Number of Dependents", 0, 10, 2, help="Number of family members financially dependent")
    with col2:
        st.markdown("##### 🎓 Education")
        edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"], help="Higher education indicates better creditworthiness")
    with col3:
        st.markdown("##### 💼 Employment")
        emp = st.selectbox("Self Employed Status", ["No", "Yes"], help="Self-employed applicants may have variable income")

with tab2:
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("##### 📈 Annual Income")
        income = st.slider("Annual Income (USD)", 200000, 10000000, 5000000, step=100000, 
                          format="$%d", help="Total yearly income from all sources")
        
        st.markdown("##### 📊 Credit Score")
        cibil = st.slider("CIBIL Score", 300, 900, 700, 
                         help="Credit score ranging from 300-900, higher is better")
    
    with col5:
        st.markdown("##### 🏦 Loan Amount")
        loan_amt = st.slider("Loan Amount Requested (USD)", 100000, 40000000, 10000000, step=100000,
                            format="$%d", help="Total loan amount requested")
        
        st.markdown("##### ⏱️ Loan Term")
        term = st.selectbox("Loan Term (Years)", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                           help="Duration of loan repayment period")

with tab3:
    st.markdown("##### 🏠 Asset Portfolio")
    st.markdown("*Provide detailed breakdown of applicant's assets*")
    
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        res = st.number_input("🏠 Residential Assets Value", 0, 30000000, 5000000, step=100000, format="%d")
        com = st.number_input("🏢 Commercial Assets Value", 0, 20000000, 2000000, step=100000, format="%d")
    with a_col2:
        lux = st.number_input("💎 Luxury Assets Value", 0, 30000000, 4000000, step=100000, format="%d")
        bank = st.number_input("🏦 Bank Assets Value", 0, 15000000, 1000000, step=100000, format="%d")
    
    # Add asset summary
    total_assets_preview = res + com + lux + bank
    st.markdown(f"""
        <div style="background: #e8f0fe; padding: 0.75rem; border-radius: 10px; margin-top: 0.5rem;">
            <strong>📊 Total Assets Value:</strong> ${total_assets_preview:,.2f}
        </div>
    """, unsafe_allow_html=True)

# --- 5. DATA PREPARATION ---
edu_enc = 0 if edu == "Graduate" else 1
emp_enc = 1 if emp == "Yes" else 0
total_assets = res + com + lux + bank

input_data = pd.DataFrame({
    'no_of_dependents': [dependents], 'education': [edu_enc], 'self_employed': [emp_enc],
    'income_annum': [income], 'loan_amount': [loan_amt], 'loan_term': [term], 'cibil_score': [cibil],
    'residential_assets_value': [res], 'commercial_assets_value': [com],
    'luxury_assets_value': [lux], 'bank_asset_value': [bank], 'total_assets': [total_assets]
})

# --- 6. PREDICTION & ANALYSIS ---
st.markdown("---")

# Centered analyze button with icon
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🚀 Analyze Application & Generate XAI Report", use_container_width=True)

if analyze_clicked:
    
    # Create two columns for results
    res_col, xai_col = st.columns([1, 1.2], gap="large")
    
    with res_col:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.markdown("### 📊 Decision Result")
        
        if prediction == 0:
            st.markdown("""
                <div class="success-card">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 48px;">✅</span>
                        <div>
                            <h2 style="margin: 0; color: #155724;">LOAN APPROVED</h2>
                            <p style="margin: 0; color: #155724;">Application meets credit criteria</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics in columns
            met1, met2 = st.columns(2)
            with met1:
                st.metric("Approval Confidence", f"{prob[0]*100:.2f}%", delta="High Confidence")
            with met2:
                st.metric("Risk Score", f"{(1-prob[0])*100:.2f}%", delta="Low Risk", delta_color="inverse")
        else:
            st.markdown("""
                <div class="error-card">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 48px;">❌</span>
                        <div>
                            <h2 style="margin: 0; color: #721c24;">LOAN REJECTED</h2>
                            <p style="margin: 0; color: #721c24;">Application does not meet credit criteria</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            met1, met2 = st.columns(2)
            with met1:
                st.metric("Risk Probability", f"{prob[1]*100:.2f}%", delta="High Risk", delta_color="normal")
            with met2:
                st.metric("Approval Likelihood", f"{prob[0]*100:.2f}%", delta="Low Chance")
        
        # Display key driver with better styling
        shap_values_local = explainer(input_data)
        top_feature_idx = np.argmax(np.abs(shap_values_local.values[0]))
        top_feature_name = input_data.columns[top_feature_idx].replace('_', ' ').title()
        
        st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 1rem; border-radius: 12px; margin-top: 1rem; border-left: 4px solid #ffc107;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">🎯</span>
                    <div>
                        <strong style="color: #856404;">Primary Decision Factor:</strong><br>
                        <span style="color: #856404; font-weight: 600;">{}</span>
                    </div>
                </div>
            </div>
        """.format(top_feature_name), unsafe_allow_html=True)
        
        # Additional applicant summary
        with st.expander("📋 Application Summary"):
            summary_data = {
                "Metric": ["Annual Income", "Loan Amount", "CIBIL Score", "Total Assets", "Loan Term"],
                "Value": [f"${income:,.0f}", f"${loan_amt:,.0f}", cibil, f"${total_assets:,.0f}", f"{term} years"]
            }
            st.table(pd.DataFrame(summary_data))

    with xai_col:
        st.markdown("### 🧠 Explainable AI Report")
        st.markdown("*SHAP Waterfall Plot - Understanding the decision*")
        
        # Create matplotlib figure with professional styling
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')
        shap.plots.waterfall(shap_values_local[0], show=False, max_display=10)
        
        # Enhance plot appearance
        ax.set_title("Feature Impact on Decision", fontsize=14, fontweight='bold', pad=20)
        ax.set_facecolor('white')
        
        st.pyplot(fig)
        
        st.markdown("""
            <div class="info-box">
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <span style="font-size: 20px;">📖</span>
                    <div style="font-size: 0.9rem;">
                        <strong>Interpretation Guide:</strong><br>
                        • Red bars push toward REJECTION<br>
                        • Blue bars push toward APPROVAL<br>
                        • Length indicates magnitude of impact
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

else:
    # Welcome message when no analysis done
    st.markdown("""
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 20px; margin: 1rem 0;">
            <span style="font-size: 48px;">🔍</span>
            <h3 style="color: #1a2c3e; margin-top: 1rem;">Ready to analyze an application?</h3>
            <p style="color: #6c757d;">Fill in the applicant details above and click the <strong>Analyze Application</strong> button<br>
            to receive an instant AI-powered credit decision with full transparency.</p>
        </div>
    """, unsafe_allow_html=True)

# --- 7. FOOTER WITH ENHANCED DESIGN ---
st.markdown("---")
st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="font-weight: 600;">🏦 LendGuard XAI Engine</span><br>
                <small style="color: #6c757d;">Powered by G-B (98.36% Accuracy)</small>
            </div>
            <div>
                <span style="font-weight: 600;">🎓 Data Science Research</span><br>
                <small style="color: #6c757d;">Explainable AI for Financial Inclusion</small>
            </div>
            <div>
                <span style="font-weight: 600;">⚡ Real-time SHAP Explanations</span><br>
                <small style="color: #6c757d;">Complete decision transparency</small>
            </div>
        </div>
        <hr style="margin: 1rem 0;">
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span>📧 Contact: zohaibdahrihere@gmail.com</span>
            <span>🔒 Data Privacy Assured</span>
            <span>📊 Version 2.0 | Enterprise Ready</span>
        </div>
    </div>
""", unsafe_allow_html=True)