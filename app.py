from huggingface_hub import hf_hub_download
import os

MODEL_PATH = "artifacts/model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("artifacts", exist_ok=True)
    hf_hub_download(
        repo_id="PSeshpavan/churn-prediction-model",
        filename="model.pkl",
        local_dir="artifacts"
    )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed", # Collapse the sidebar by default
)

# --- Premium Custom Styling (Dark/Modern Tech Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title Banner Styling */
    .title-banner {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 12px;
        margin-top: 25px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    
    .title-banner h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 3px;
    }
    
    .title-banner p {
        font-size: 0.85rem;
        color: #9ca3af;
        margin: 0;
    }

    /* Container for Form Inputs */
    .form-container {
        background: rgba(31, 41, 55, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    
    /* Status Meters */
    .status-container {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 5px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .high-risk-bg {
        background: radial-gradient(circle, rgba(239, 68, 68, 0.15) 0%, rgba(17, 24, 39, 0.7) 100%);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .medium-risk-bg {
        background: radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, rgba(17, 24, 39, 0.7) 100%);
        border-color: rgba(245, 158, 11, 0.35);
    }
    
    .low-risk-bg {
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, rgba(17, 24, 39, 0.7) 100%);
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .status-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2px 0;
    }
    
    .status-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    /* Shrink Streamlit block padding globally to fit on screen */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
    }
    .element-container {
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Description Banner ---
st.markdown("""
<div class="title-banner">
    <h2 style="font-size: 1.4rem; margin: 0; padding: 0; color: #f3f4f6; font-weight: 700;">🏦 Customer Churn Prediction Portal</h2>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join("artifacts", "model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model()

if not pipeline:
    st.error("Model pipeline not found! Please ensure 'artifacts/model.pkl' exists.")
    st.stop()

# --- Main Page Form Configuration ---
with st.container():
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        is_active_member = st.selectbox("Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=40, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])

    with c3:
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with c4:
        balance = st.number_input("Balance ($)", min_value=0.0, max_value=250000.0, value=60000.0, step=1000.0)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)

    with c5:
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
        num_of_products = st.selectbox("Num Products", [1, 2, 3, 4], index=1)

# Package feature inputs
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# --- Prediction & Analytics Section ---
col_risk, col_explain = st.columns([2, 3])

# Run predictions and SHAP explainability inside the spinner loading animation
with st.spinner("Analyzing customer profile & running model..."):
    # Calculate prediction
    prediction_prob = pipeline.predict_proba(input_df)[0][1]
    is_high_risk = prediction_prob > 0.5
    
    # Calculate SHAP values
    shap_df = None
    shap_error = None
    fig = None
    try:
        preprocessor = pipeline.named_steps['prep']
        model = pipeline.named_steps['clf']
        
        # Preprocess the input
        X_transformed = preprocessor.transform(input_df)
        
        # Reconstruct feature names
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(numeric_features) + list(categorical_features)
        
        # Compute SHAP values on the fly at runtime (explainer is not loaded from disk)
        try:
            bg_data = pd.read_csv("data.csv")
            X_bg = bg_data.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"], errors="ignore")
            # Sample 100 rows for background reference
            X_bg_sample = X_bg.sample(100, random_state=42)
            X_bg_transformed = preprocessor.transform(X_bg_sample)
            
            explainer = shap.TreeExplainer(model, data=X_bg_transformed, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X_transformed)
        except Exception as explainer_err:
            # Fallback if background data loading/explainer initialization fails
            explainer = shap.TreeExplainer(model)
            try:
                shap_values = explainer.shap_values(X_transformed, check_additivity=False)
            except Exception:
                shap_values = explainer.shap_values(X_transformed)
        
        # Robust extraction of 1D SHAP values for class 1
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
            
        # If shape is 3D
        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                if shap_vals.shape[2] == 2:
                    shap_vals = shap_vals[0, :, 1]
                elif shap_vals.shape[0] == 2:
                    shap_vals = shap_vals[1, 0, :]
            elif shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
            elif shap_vals.ndim == 1:
                pass
        
        # Create Pandas summary
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': shap_vals,
            'Absolute': np.abs(shap_vals)
        }).sort_values(by='Absolute', ascending=False).head(5)
        
        # Plot contributions
        fig, ax = plt.subplots(figsize=(6, 2.3), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        
        colors = ['#ef4444' if val > 0 else '#10b981' for val in shap_df['Contribution']]
        bars = ax.barh(shap_df['Feature'][::-1], shap_df['Contribution'][::-1], color=colors[::-1], height=0.6)
        
        # Format Chart Axes & Labels
        ax.tick_params(colors='#c9d1d9', labelsize=8)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#374151')
        
        ax.axvline(x=0, color='#9ca3af', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Feature Influence (SHAP)", color='#c9d1d9', fontsize=8)
        plt.tight_layout()
    except Exception as e:
        shap_error = str(e)

# Render results in parallel columns
with col_risk:
    st.markdown("### 📊 Prediction Result")
    
    # Styled Meter Card
    if prediction_prob >= 0.6:
        bg_class = "high-risk-bg"
        color = "#ef4444"
        risk_label = "HIGH RISK"
    elif prediction_prob >= 0.3:
        bg_class = "medium-risk-bg"
        color = "#f59e0b"
        risk_label = "MEDIUM RISK"
    else:
        bg_class = "low-risk-bg"
        color = "#10b981"
        risk_label = "LOW RISK"
        
    st.markdown(f"""
    <div class="status-container {bg_class}">
        <div class="status-label">Churn Risk Probability</div>
        <div class="status-text" style="color: {color};">{prediction_prob * 100:.1f}%</div>
        <div style="font-weight: 600; letter-spacing: 0.05em; color: {color};">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_explain:
    st.markdown("### 🧠 Explainability Graph (SHAP)")
    if shap_df is not None and fig is not None:
        st.pyplot(fig)
    else:
        st.warning(f"Could not compute SHAP values for the current model. Error: {shap_error}")
