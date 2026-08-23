"""
evaluate.py
------------
Loads the trained model and test data, prints classification metrics,
and generates a SHAP explainability plot.

Run directly:  python src/evaluate.py
"""

import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/fraud_model.pkl"


def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_test, y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("docs/confusion_matrix.png")
    print("Saved confusion matrix to docs/confusion_matrix.png")


def explain(model, X_test):
    """Generate a SHAP summary plot showing which features drive fraud predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:500])  # sample for speed

    shap.summary_plot(shap_values, X_test.iloc[:500], show=False)
    plt.savefig("docs/shap_summary.png", bbox_inches="tight")
    print("Saved SHAP summary plot to docs/shap_summary.png")


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()
    evaluate(model, X_test, y_test)
    explain(model, X_test)