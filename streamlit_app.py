"""
streamlit_app.py
-----------------
Interactive dashboard: "From Charts to Chips: AI's Role in Shaping
Smarter Healthcare Governance"

Lets a user:
    - Enter hospital operational data via a form
    - Get an instant Governance Risk prediction with confidence scores
    - View AI-generated (rule-based, explainable) recommendations
    - Explore dataset-level visualizations & model comparison charts
      (correlation heatmap, class distribution, feature importance,
      confusion matrix, ROC curve, model comparison)

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Make src/ importable regardless of the working directory streamlit is
# launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from predict import load_best_model, predict_governance_risk, generate_recommendations  # noqa: E402
from preprocessing import load_data  # noqa: E402

# ----------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smarter Healthcare Governance — AI Dashboard",
    page_icon="🏥",
    layout="wide",
)

RISK_COLORS = {
    "Efficient Governance": "#2E7D32",
    "Moderate Risk": "#F9A825",
    "High Governance Risk": "#C62828",
}

st.markdown(
    """
    <style>
    .big-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
    .subtitle {font-size: 1.05rem; color: #6b7280; margin-top: 0;}
    .risk-badge {
        padding: 0.6rem 1.2rem; border-radius: 10px; color: white;
        font-size: 1.3rem; font-weight: 700; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">🏥 From Charts to Chips</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI\'s Role in Shaping Smarter Healthcare Governance — '
    'a decision-support dashboard for hospital administrators & policymakers, '
    'not a disease-diagnosis tool.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Governance Risk", "📊 Data & Model Insights"],
)

model, metadata = load_best_model()
class_names = metadata["class_names"]
best_model_name = metadata["best_model_name"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active model:** {best_model_name}")
st.sidebar.markdown(f"**Macro F1 Score:** {metadata['metrics']['F1 Score']:.3f}")
st.sidebar.markdown(f"**Accuracy:** {metadata['metrics']['Accuracy']:.3f}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "This tool predicts *hospital administrative/governance risk* "
    "(efficiency, resourcing, safety-process indicators) — it does not "
    "diagnose or predict disease in individual patients."
)

# ========================================================================
# PAGE 1 — Prediction
# ========================================================================
if page == "🔮 Predict Governance Risk":
    st.subheader("Enter Hospital Operational Data")

    with st.form("hospital_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            hospital_type = st.selectbox("Hospital Type", ["Government", "Private", "Trust/NGO"])
            location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])
            num_patients = st.number_input("Number of Patients", min_value=1, max_value=5000, value=500)
            avg_wait = st.slider("Average Waiting Time (minutes)", 0, 180, 45)

        with col2:
            bed_occupancy = st.slider("Bed Occupancy Rate (%)", 0, 100, 70)
            num_doctors = st.number_input("Number of Doctors", min_value=1, max_value=500, value=40)
            nurse_ratio = st.slider("Nurse-to-Patient Ratio", 0.0, 1.0, 0.35, step=0.01)
            budget = st.number_input("Hospital Budget (₹ Crore)", min_value=1.0, max_value=300.0, value=50.0)

        with col3:
            equipment = st.slider("Medical Equipment Availability (%)", 0, 100, 75)
            satisfaction = st.slider("Patient Satisfaction Score (0-10)", 0.0, 10.0, 6.5, step=0.1)
            emergency_cases = st.number_input("Emergency Cases (monthly)", min_value=0, max_value=1000, value=60)
            infection_rate = st.slider("Infection Rate (%)", 0.0, 25.0, 5.0, step=0.1)
            readmission_rate = st.slider("Readmission Rate (%)", 0.0, 40.0, 10.0, step=0.1)

        submitted = st.form_submit_button("🔍 Predict Governance Risk", use_container_width=True)

    if submitted:
        raw_record = {
            "Hospital_Type": hospital_type,
            "Location": location,
            "Num_Patients": num_patients,
            "Avg_Waiting_Time_Min": avg_wait,
            "Bed_Occupancy_Rate": bed_occupancy,
            "Num_Doctors": num_doctors,
            "Nurse_to_Patient_Ratio": nurse_ratio,
            "Hospital_Budget_Crore": budget,
            "Medical_Equipment_Availability": equipment,
            "Patient_Satisfaction_Score": satisfaction,
            "Emergency_Cases": emergency_cases,
            "Infection_Rate": infection_rate,
            "Readmission_Rate": readmission_rate,
        }

        predicted_label, confidence = predict_governance_risk(raw_record)
        color = RISK_COLORS.get(predicted_label, "#374151")

        st.markdown("### Prediction Result")
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(
                f'<div class="risk-badge" style="background-color:{color};">'
                f'{predicted_label}</div>',
                unsafe_allow_html=True,
            )
            st.metric("Prediction Confidence", f"{max(confidence.values()):.1%}")

        with r2:
            conf_df = pd.DataFrame({
                "Governance Class": list(confidence.keys()),
                "Confidence": list(confidence.values()),
            }).sort_values("Confidence", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 2.8))
            bar_colors = [RISK_COLORS.get(c, "#374151") for c in conf_df["Governance Class"]]
            ax.barh(conf_df["Governance Class"], conf_df["Confidence"], color=bar_colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            for i, v in enumerate(conf_df["Confidence"]):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("### 🤖 AI-Generated Governance Recommendations")
        recs = generate_recommendations(raw_record, predicted_label)
        st.info(recs[0])
        for r in recs[1:]:
            st.markdown(f"- {r}")

# ========================================================================
# PAGE 2 — Data & Model Insights
# ========================================================================
else:
    st.subheader("Dataset & Model Insights")

    tab1, tab2, tab3 = st.tabs(["📈 Dataset Overview", "🧠 Model Performance", "🔍 Feature Importance"])

    with tab1:
        df = load_data()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class Distribution**")
            img_path = os.path.join(VISUALS_DIR, "class_distribution.png")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
        with c2:
            st.markdown("**Correlation Heatmap**")
            img_path = os.path.join(VISUALS_DIR, "correlation_heatmap.png")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

        st.markdown("**Sample of the Dataset**")
        st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        results_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            st.markdown("**Model Comparison Table**")
            st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                    "Recall": "{:.3f}", "F1 Score": "{:.3f}",
                }).highlight_max(subset=["F1 Score"], color="#c6f6d5"),
                use_container_width=True,
            )

        img_path = os.path.join(VISUALS_DIR, "model_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

        st.markdown(f"**Confusion Matrix & ROC Curve — {best_model_name} (best model)**")
        cm_path = os.path.join(VISUALS_DIR, f"confusion_matrix_{best_model_name.replace(' ', '_')}.png")
        roc_path = os.path.join(VISUALS_DIR, f"roc_curve_{best_model_name.replace(' ', '_')}.png")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
        with c2:
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)

        with st.expander("View confusion matrix & ROC curve for all models"):
            model_names = results_df["Model"].tolist() if os.path.exists(results_path) else []
            for m in model_names:
                st.markdown(f"**{m}**")
                cc1, cc2 = st.columns(2)
                cm_p = os.path.join(VISUALS_DIR, f"confusion_matrix_{m.replace(' ', '_')}.png")
                roc_p = os.path.join(VISUALS_DIR, f"roc_curve_{m.replace(' ', '_')}.png")
                with cc1:
                    if os.path.exists(cm_p):
                        st.image(cm_p, use_container_width=True)
                with cc2:
                    if os.path.exists(roc_p):
                        st.image(roc_p, use_container_width=True)

    with tab3:
        st.markdown(f"**Feature Importance — {best_model_name}**")
        fi_path = os.path.join(VISUALS_DIR, f"feature_importance_{best_model_name.replace(' ', '_')}.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.caption("Feature importance is only available for tree-based models.")

st.divider()
st.caption(
    "College project demonstration — synthetic data used to illustrate an "
    "end-to-end ML pipeline for hospital governance decision support. "
    "Not intended for real clinical or administrative deployment."
)
