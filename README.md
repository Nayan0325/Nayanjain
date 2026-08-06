# From Charts to Chips: AI's Role in Shaping Smarter Healthcare Governance

An end-to-end **machine learning application** that demonstrates how AI can
support **hospital governance and administrative decision-making** —
predicting whether a hospital's operational profile indicates *Efficient
Governance*, *Moderate Risk*, or *High Governance Risk*, and generating
explainable, actionable recommendations.

> ⚠️ **Scope note:** This project is about hospital **governance/operations**
> (staffing, budgets, wait times, infection & readmission rates, etc.) — it
> is **not** a disease-diagnosis tool and makes no clinical predictions
> about individual patients.

---

## 1. Project Structure

```
healthcare_governance_ai/
├── data/
│   ├── generate_dataset.py          # synthetic dataset generator
│   └── healthcare_governance_data.csv
├── src/
│   ├── preprocessing.py             # cleaning, encoding, scaling, splitting
│   ├── train_models.py              # trains/evaluates/compares 4 models
│   └── predict.py                   # inference + recommendation engine
├── app/
│   └── streamlit_app.py             # interactive dashboard
├── models/                          # saved model + preprocessing artifacts (generated)
├── visuals/                         # saved charts (generated)
├── requirements.txt
└── README.md
```

## 2. Dataset

Because real hospital-level governance datasets are not publicly available
(privacy/regulatory restrictions), `data/generate_dataset.py` builds a
**realistic synthetic dataset** (2,000 hospitals) using domain-informed
statistical rules based on typical hospital-administration benchmarks, with
injected missing values to demonstrate real-world data cleaning.

**Features:**
`Hospital_Type`, `Location`, `Num_Patients`, `Avg_Waiting_Time_Min`,
`Bed_Occupancy_Rate`, `Num_Doctors`, `Nurse_to_Patient_Ratio`,
`Hospital_Budget_Crore`, `Medical_Equipment_Availability`,
`Patient_Satisfaction_Score`, `Emergency_Cases`, `Infection_Rate`,
`Readmission_Rate`

**Target (`Governance_Risk`):** `Efficient Governance` / `Moderate Risk` /
`High Governance Risk` — derived from a weighted governance-health score
(doctor/nurse staffing, budget, equipment, satisfaction vs. wait time,
occupancy imbalance, infection & readmission rates) plus random noise, then
split into class-balanced tertiles.

## 3. Pipeline

1. **Preprocessing** (`src/preprocessing.py`)
   - Median imputation (numeric) / mode imputation (categorical) for
     missing values
   - One-hot encoding for `Hospital_Type`, `Location`
   - Label encoding for the target
   - `StandardScaler` for numeric features
   - Stratified 80/20 train-test split
   - Fitted scaler/encoders saved to `models/` for reuse at inference time

2. **Model Training & Comparison** (`src/train_models.py`)
   - Trains **Logistic Regression**, **Decision Tree**, **Random Forest**,
     **XGBoost**
   - Evaluates each with Accuracy, macro Precision/Recall/F1, Confusion
     Matrix, and One-vs-Rest multi-class ROC curves
   - Saves comparison charts + per-model confusion matrices/ROC
     curves/feature importances to `visuals/`
   - Selects the best model by macro F1 score and pickles it to
     `models/best_model.pkl`

   **Latest run results** (your numbers may vary slightly with the random
   seed):

   | Model | Accuracy | Precision | Recall | F1 Score |
   |---|---|---|---|---|
   | XGBoost | 0.765 | 0.765 | 0.764 | 0.764 |
   | Random Forest | 0.750 | 0.747 | 0.749 | 0.748 |
   | Logistic Regression | 0.715 | 0.711 | 0.714 | 0.711 |
   | Decision Tree | 0.673 | 0.665 | 0.671 | 0.666 |

3. **Prediction & Recommendations** (`src/predict.py`)
   - Loads the saved model + preprocessing artifacts
   - `predict_governance_risk(raw_record)` → predicted class + confidence
     scores per class
   - `generate_recommendations(raw_record, predicted_label)` → a
     transparent, rule-based set of governance recommendations (kept
     rule-based rather than a second opaque model, so every
     recommendation is explainable to a hospital administrator)

4. **Dashboard** (`app/streamlit_app.py`)
   - **Predict Governance Risk** page: form to enter hospital metrics →
     instant prediction, confidence bar chart, and AI-generated
     recommendations
   - **Data & Model Insights** page: dataset overview, correlation
     heatmap, class distribution, model comparison table/chart, confusion
     matrices, ROC curves, and feature importance

## 4. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python data/generate_dataset.py

# 3. Train & evaluate all models (saves best model + visuals)
python src/train_models.py

# 4. (Optional) quick CLI prediction sanity check
python src/predict.py

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

## 5. Notes for Presentation

- The project intentionally separates **preprocessing**, **training**,
  **prediction**, and the **UI** into different files — a standard
  production-style ML project layout.
- The recommendation engine is rule-based for **explainability**: a
  governance tool that flags "why" a hospital is at risk (and what to do
  about it) is more useful — and more trustworthy — for administrators
  than a black-box label alone.
- Because the underlying data is synthetic, the emphasis of this project
  is the **end-to-end AI pipeline and decision-support workflow**, which
  generalizes directly to real hospital data if/when it becomes available.
