"""
train_model.py
----------------
Trains multiple classifiers (XGBoost, Random Forest, LightGBM) on the
preprocessed data, compares them on the test set, and saves the best
performing model to models/fraud_model.pkl.

Run directly:  python src/train_model.py
(Run preprocessing.py first!)
"""

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/fraud_model.pkl"
COMPARISON_PATH = "docs/model_comparison.csv"


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_models():
    """
    Three models chosen to represent different algorithm families:
    - XGBoost: gradient boosting, industry standard for tabular fraud data
    - Random Forest: bagging ensemble, more robust to overfitting, less tuning needed
    - LightGBM: gradient boosting like XGBoost, but faster on large datasets
      via histogram-based splitting
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1,
        ),
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}

    for name, model in get_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision (fraud)": round(precision_score(y_test, y_pred), 4),
            "Recall (fraud)": round(recall_score(y_test, y_pred), 4),
            "F1 (fraud)": round(f1_score(y_test, y_pred), 4),
        })

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    return results_df, trained_models


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed()
    results_df, trained_models = train_and_compare(X_train, X_test, y_train, y_test)

    results_df.to_csv(COMPARISON_PATH, index=False)
    print(f"\nSaved comparison table to {COMPARISON_PATH}")

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']})")
    print(f"Saved best model to {MODEL_PATH}")