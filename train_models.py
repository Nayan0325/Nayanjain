"""
train_models.py
----------------
Trains and compares multiple ML models for Hospital Governance Risk
classification:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

For each model we compute Accuracy, Precision, Recall, F1 Score
(macro-averaged, since this is a multi-class problem), a Confusion
Matrix, and a multi-class ROC curve (One-vs-Rest).

The best model (highest macro F1 score) is saved as models/best_model.pkl
along with a metadata file recording which algorithm won and why.

Visualizations produced (saved into visuals/):
    - correlation_heatmap.png
    - class_distribution.png
    - model_comparison.png
    - confusion_matrix_<model>.png (for every model)
    - roc_curve_<model>.png (for every model)
    - feature_importance_<model>.png (for tree-based models)

Run:
    python src/train_models.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from preprocessing import preprocess_pipeline, load_data, NUMERIC_COLS, TARGET_COL

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]


# ----------------------------------------------------------------------
# Exploratory visualizations (run once, on raw data)
# ----------------------------------------------------------------------
def plot_correlation_heatmap():
    df = load_data()
    corr = df[NUMERIC_COLS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Hospital Governance Features", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_class_distribution():
    df = load_data()
    plt.figure(figsize=(7, 5))
    order = df[TARGET_COL].value_counts().index
    sns.countplot(data=df, x=TARGET_COL, order=order, palette=PALETTE)
    plt.title("Governance Risk Class Distribution", fontsize=13, weight="bold")
    plt.xlabel("Governance Risk Category")
    plt.ylabel("Number of Hospitals")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "class_distribution.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Per-model evaluation visualizations
# ----------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred, class_names, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, weight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(VISUALS_DIR, fname), dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, class_names, model_name):
    """One-vs-Rest multi-class ROC curve."""
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    n_classes = len(class_names)

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})",
                 color=PALETTE[i % len(PALETTE)], linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (One-vs-Rest) — {model_name}", fontsize=12, weight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fname = f"roc_curve_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(VISUALS_DIR, fname), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:12]  # top 12
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], color="#2E86AB")
    plt.title(f"Top Feature Importances — {model_name}", fontsize=12, weight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    fname = f"feature_importance_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(VISUALS_DIR, fname), dpi=150)
    plt.close()


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(results_df))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width, label=metric, color=PALETTE[i])

    plt.xticks(x + width * 1.5, results_df["Model"], rotation=10)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison — Governance Risk Classification", fontsize=13, weight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def get_model_zoo():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )
    return models


def train_and_evaluate():
    print("=" * 70)
    print("STEP 1: Exploratory visualizations")
    print("=" * 70)
    plot_correlation_heatmap()
    plot_class_distribution()
    print("Saved correlation_heatmap.png and class_distribution.png")

    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing")
    print("=" * 70)
    X_train, X_test, y_train, y_test, class_names, feature_names = preprocess_pipeline()
    print(f"Classes: {class_names}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 3: Training & evaluating models")
    print("=" * 70)

    models = get_model_zoo()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1 Score": f1,
        })
        trained_models[name] = model

        plot_confusion_matrix(y_test, y_pred, class_names, name)
        plot_roc_curve(y_test, y_proba, class_names, name)
        plot_feature_importance(model, feature_names, name)

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("STEP 4: Model comparison")
    print("=" * 70)
    print(results_df.to_string(index=False))
    plot_model_comparison(results_df)

    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison_results.csv"), index=False)

    # ---- Select and save best model ----
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name} (F1 Score = {results_df.iloc[0]['F1 Score']:.4f})")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "best_model_name": best_model_name,
        "class_names": class_names,
        "feature_names": feature_names,
        "metrics": results_df.iloc[0].to_dict(),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved best model to models/best_model.pkl")
    print(f"Saved metadata to models/model_metadata.pkl")
    print(f"All visualizations saved to visuals/")

    return results_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate()
