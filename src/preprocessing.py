"""
preprocessing.py
-----------------
Loads raw transaction data, cleans it, scales features, handles
class imbalance with SMOTE, and saves train/test splits to disk.

Run directly:  python src/preprocessing.py
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

RAW_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from Kaggle and place it there first."
        )
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        print("Filled missing numeric values with column median")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale 'Amount' and 'Time' columns (the rest of the columns in the
    Kaggle credit card dataset are already PCA-transformed/scaled).
    """
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return df


def split_and_balance(df: pd.DataFrame, target_col: str = "Class"):
    """
    Split into train/test (stratified so both sets keep the same
    fraud ratio), then apply SMOTE ONLY to the training set.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Before SMOTE -> fraud cases in train: {y_train.sum()} / {len(y_train)}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  -> fraud cases in train: {y_train_res.sum()} / {len(y_train_res)}")

    return X_train_res, X_test, y_train_res, y_test


def save_processed(X_train, X_test, y_train, y_test, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    print(f"Saved processed splits to {out_dir}/")


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)
    save_processed(X_train, X_test, y_train, y_test, PROCESSED_DIR)