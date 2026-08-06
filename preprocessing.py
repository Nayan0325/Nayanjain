"""
preprocessing.py
-----------------
Handles all data preprocessing for the Healthcare Governance Risk
classification project:
    1. Load raw CSV data
    2. Handle missing values (median imputation for numeric columns)
    3. Encode categorical variables (Hospital_Type, Location -> One-Hot;
       Governance_Risk target -> Label Encoding)
    4. Feature scaling (StandardScaler) for numerical columns
    5. Train/test split
    6. Persist the fitted scaler + label encoder + feature column order
       to disk (models/) so the Streamlit app can reuse the exact same
       transformations at inference time.

This module can be imported by train_models.py and by the Streamlit app.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "healthcare_governance_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_COL = "Governance_Risk"
ID_COL = "Hospital_ID"
CATEGORICAL_COLS = ["Hospital_Type", "Location"]

# The core numeric governance features requested in the project spec
NUMERIC_COLS = [
    "Num_Patients",
    "Avg_Waiting_Time_Min",
    "Bed_Occupancy_Rate",
    "Num_Doctors",
    "Nurse_to_Patient_Ratio",
    "Hospital_Budget_Crore",
    "Medical_Equipment_Availability",
    "Patient_Satisfaction_Score",
    "Emergency_Cases",
    "Infection_Rate",
    "Readmission_Rate",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values with the column median (robust to
    outliers) and missing categorical values with the column mode.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    One-hot encode categorical predictor columns and label-encode the
    target column.

    Parameters
    ----------
    df : DataFrame containing raw (but missing-value-free) data.
    fit : If True, fit new encoders (training time). If False, reuse the
          encoders dict passed in (inference time).
    encoders : dict with keys 'label_encoder' and 'onehot_columns' used
               when fit=False.

    Returns
    -------
    X_encoded : DataFrame of encoded predictor features
    y_encoded : np.ndarray of encoded target labels (or None if target
                column absent, e.g. at inference time)
    encoders : dict of fitted encoding artifacts
    """
    df = df.copy()

    # ---- Predictors: One-hot encode categorical columns ----
    X = df.drop(columns=[c for c in [TARGET_COL, ID_COL] if c in df.columns])
    X_encoded = pd.get_dummies(X, columns=[c for c in CATEGORICAL_COLS if c in X.columns])

    if fit:
        onehot_columns = X_encoded.columns.tolist()
    else:
        onehot_columns = encoders["onehot_columns"]
        # Ensure inference-time data has exactly the same dummy columns
        for col in onehot_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[onehot_columns]

    # ---- Target: Label encode ----
    y_encoded = None
    label_encoder = encoders["label_encoder"] if (encoders and "label_encoder" in encoders) else None
    if TARGET_COL in df.columns:
        if fit:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(df[TARGET_COL])
        else:
            y_encoded = label_encoder.transform(df[TARGET_COL])

    new_encoders = {
        "label_encoder": label_encoder,
        "onehot_columns": onehot_columns,
    }
    return X_encoded, y_encoded, new_encoders


def scale_features(X: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    """Standardize numeric feature columns (zero mean, unit variance)."""
    X = X.copy()
    numeric_present = [c for c in NUMERIC_COLS if c in X.columns]

    if fit:
        scaler = StandardScaler()
        X[numeric_present] = scaler.fit_transform(X[numeric_present])
    else:
        X[numeric_present] = scaler.transform(X[numeric_present])

    return X, scaler


def preprocess_pipeline(save_artifacts: bool = True, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline used at TRAINING time:
        load -> clean -> encode -> split -> scale -> (save artifacts)

    Returns X_train, X_test, y_train, y_test, class_names, feature_names
    """
    df = load_data()
    df = handle_missing_values(df)

    X_encoded, y_encoded, encoders = encode_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_test_scaled, _ = scale_features(X_test, fit=False, scaler=scaler)

    class_names = encoders["label_encoder"].classes_.tolist()
    feature_names = X_encoded.columns.tolist()

    if save_artifacts:
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
            pickle.dump(feature_names, f)

    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, feature_names


def load_inference_artifacts():
    """Load the scaler, encoders, and feature_names saved at training time."""
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return scaler, encoders, feature_names


def preprocess_single_record(raw_record: dict):
    """
    Preprocess a SINGLE new hospital record (e.g. from the Streamlit
    form) using the artifacts saved during training. Used at inference
    time in predict.py / streamlit_app.py.

    raw_record: dict of {feature_name: value} matching the raw dataset
                schema (Hospital_Type, Location, and the numeric cols).
    Returns a scaled, encoded 1-row DataFrame ready for model.predict().
    """
    scaler, encoders, feature_names = load_inference_artifacts()

    df = pd.DataFrame([raw_record])
    df = handle_missing_values_for_inference(df)

    X_encoded, _, _ = encode_features(df, fit=False, encoders=encoders)
    X_scaled, _ = scale_features(X_encoded, fit=False, scaler=scaler)

    return X_scaled


def handle_missing_values_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight missing-value guard for single-row inference input."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0)
    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_names, feature_names = preprocess_pipeline()
    print("Preprocessing complete.")
    print("Classes:", class_names)
    print("Num features:", len(feature_names))
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
