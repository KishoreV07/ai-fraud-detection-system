"""
train_model.py
----------------
Trains an XGBoost classifier on the preprocessed data and saves it
to models/fraud_model.pkl.

Run directly:  python src/train_model.py
(Run preprocessing.py first!)
"""

import pandas as pd
import joblib
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/fraud_model.pkl"


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train(X_train, y_train) -> XGBClassifier:
    """
    XGBoost is chosen as the primary model because it consistently
    performs best on tabular, imbalanced fraud data in industry
    benchmarks, handles non-linear feature interactions well, and
    trains fast even on 100K+ rows.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()
    model = train(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")