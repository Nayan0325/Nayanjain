"""
predict.py
----------
Loads the saved best model + preprocessing artifacts and provides:
    1. predict_governance_risk(raw_record) -> (label, confidence_dict)
    2. generate_recommendations(raw_record, predicted_label) -> list[str]

'raw_record' is a plain dict matching the raw dataset schema, e.g.:
    {
        "Hospital_Type": "Government",
        "Location": "Urban",
        "Num_Patients": 500,
        "Avg_Waiting_Time_Min": 40,
        "Bed_Occupancy_Rate": 78,
        "Num_Doctors": 45,
        "Nurse_to_Patient_Ratio": 0.4,
        "Hospital_Budget_Crore": 60,
        "Medical_Equipment_Availability": 80,
        "Patient_Satisfaction_Score": 7.2,
        "Emergency_Cases": 60,
        "Infection_Rate": 4.5,
        "Readmission_Rate": 9.0,
    }

This module is used both by the Streamlit app and can be run standalone
for a quick command-line sanity check.
"""

import os
import pickle

import numpy as np

from preprocessing import preprocess_single_record

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_best_model():
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return model, metadata


def predict_governance_risk(raw_record: dict):
    """
    Predict the governance-risk class for a single hospital record.

    Returns
    -------
    predicted_label : str
    confidence_scores : dict {class_name: probability}
    """
    model, metadata = load_best_model()
    class_names = metadata["class_names"]

    X_scaled = preprocess_single_record(raw_record)
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label = class_names[pred_idx]

    confidence_scores = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    return predicted_label, confidence_scores


# ----------------------------------------------------------------------
# Rule-based "AI-generated" recommendation engine.
#
# This uses the same domain knowledge that shaped the synthetic target
# variable to translate a hospital's raw metrics into actionable
# governance guidance. It is intentionally transparent/rule-based
# (rather than another opaque model) so recommendations are explainable
# — an important property for real governance/decision-support tools.
# ----------------------------------------------------------------------
def generate_recommendations(raw_record: dict, predicted_label: str) -> list:
    recs = []

    doctors = raw_record.get("Num_Doctors", 0)
    patients = raw_record.get("Num_Patients", 1) or 1
    doc_ratio = doctors / patients

    if doc_ratio < 0.03:
        recs.append(
            "Doctor-to-patient ratio is low. Consider recruiting additional "
            "medical staff or redistributing doctors from lower-demand departments."
        )

    if raw_record.get("Nurse_to_Patient_Ratio", 1) < 0.25:
        recs.append(
            "Nurse-to-patient ratio is below recommended levels. Increasing "
            "nursing staff can reduce patient wait times and improve care quality."
        )

    if raw_record.get("Avg_Waiting_Time_Min", 0) > 60:
        recs.append(
            "Average waiting time is high. Streamline patient triage, "
            "expand outpatient scheduling slots, or introduce a digital "
            "queue-management system."
        )

    occupancy = raw_record.get("Bed_Occupancy_Rate", 0)
    if occupancy > 90:
        recs.append(
            "Bed occupancy is critically high, risking overcrowding. "
            "Consider capacity expansion, faster discharge protocols, or "
            "patient transfer partnerships with nearby facilities."
        )
    elif occupancy < 50:
        recs.append(
            "Bed occupancy is low relative to capacity, suggesting "
            "underutilized resources. Review staffing/budget allocation "
            "for potential reallocation to higher-demand units."
        )

    if raw_record.get("Medical_Equipment_Availability", 100) < 60:
        recs.append(
            "Medical equipment availability is low. Prioritize budget "
            "toward procurement/maintenance of essential diagnostic and "
            "treatment equipment."
        )

    if raw_record.get("Infection_Rate", 0) > 8:
        recs.append(
            "Infection rate is above safe thresholds. Strengthen infection-"
            "control protocols, sanitation audits, and staff training on "
            "hygiene compliance."
        )

    if raw_record.get("Readmission_Rate", 0) > 15:
        recs.append(
            "Readmission rate is high, which may indicate gaps in "
            "discharge planning or post-treatment follow-up. Introduce "
            "structured discharge summaries and follow-up call programs."
        )

    if raw_record.get("Patient_Satisfaction_Score", 10) < 5:
        recs.append(
            "Patient satisfaction score is low. Conduct patient feedback "
            "surveys and address service-quality and communication gaps."
        )

    if raw_record.get("Hospital_Budget_Crore", 100) < 20 and predicted_label != "Efficient Governance":
        recs.append(
            "Budget levels appear constrained relative to patient load. "
            "Explore government health scheme funding, CSR partnerships, "
            "or public-private collaboration models."
        )

    if not recs:
        recs.append(
            "Current governance indicators are within healthy ranges. "
            "Maintain regular audits and continue monitoring key metrics "
            "to sustain efficient governance."
        )

    # Prepend an overall summary line based on the predicted class
    summary_map = {
        "Efficient Governance": (
            "Overall assessment: This hospital demonstrates EFFICIENT "
            "governance. Focus on sustaining current practices."
        ),
        "Moderate Risk": (
            "Overall assessment: This hospital shows MODERATE governance "
            "risk. Targeted improvements below can shift it toward "
            "efficient governance."
        ),
        "High Governance Risk": (
            "Overall assessment: This hospital is at HIGH governance risk. "
            "Prioritized intervention is recommended across the flagged "
            "areas below."
        ),
    }
    recs.insert(0, summary_map.get(predicted_label, ""))

    return recs


if __name__ == "__main__":
    sample = {
        "Hospital_Type": "Government",
        "Location": "Rural",
        "Num_Patients": 800,
        "Avg_Waiting_Time_Min": 75,
        "Bed_Occupancy_Rate": 95,
        "Num_Doctors": 15,
        "Nurse_to_Patient_Ratio": 0.15,
        "Hospital_Budget_Crore": 12,
        "Medical_Equipment_Availability": 45,
        "Patient_Satisfaction_Score": 4.0,
        "Emergency_Cases": 120,
        "Infection_Rate": 12.0,
        "Readmission_Rate": 22.0,
    }
    label, confidence = predict_governance_risk(sample)
    print("Predicted governance risk:", label)
    print("Confidence scores:")
    for k, v in confidence.items():
        print(f"  {k}: {v:.2%}")
    print("\nRecommendations:")
    for r in generate_recommendations(sample, label):
        print(" -", r)
