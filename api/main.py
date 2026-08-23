"""
main.py — FastAPI backend
---------------------------
Serves the trained fraud detection model as a REST API and logs
every prediction to a SQLite database.

Run:  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

from api.database import init_db, save_prediction, get_history

app = FastAPI(title="AI Fraud Detection API")

init_db()  # ensure table exists on startup

try:
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    model = None
    scaler = None


class Transaction(BaseModel):
    Amount: float
    Time: float
    features: dict


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run training pipeline first.",
        )

    row = transaction.features.copy()
    row["scaled_amount"] = scaler.transform([[transaction.Amount]])[0][0]
    row["scaled_time"] = scaler.transform([[transaction.Time]])[0][0]

    df = pd.DataFrame([row])
    df = df[model.get_booster().feature_names]

    fraud_probability = float(model.predict_proba(df)[0][1])
    is_fraud = fraud_probability > 0.5
    risk_level = (
        "High" if fraud_probability > 0.7
        else "Medium" if fraud_probability > 0.3
        else "Low"
    )

    save_prediction(
        amount=transaction.Amount,
        time_val=transaction.Time,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        risk_level=risk_level,
    )

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


@app.get("/history")
def history(limit: int = 100):
    return get_history(limit)