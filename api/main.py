"""
main.py — FastAPI backend (hardened)
---------------------------------------
Serves the trained fraud detection model as a REST API. Includes:
- Input validation (required features, sane ranges)
- Structured logging
- Graceful error handling (never crashes on bad input)
- CORS support (so the frontend can call this from any origin)
- Batch size limits (prevents resource exhaustion)
- A real health check that reports model status

Run:  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import logging
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd

from api.database import init_db, save_prediction, get_history

# ---------------------------------------------------------------------------
# Logging setup — every prediction and error gets logged with a timestamp.
# In production this would go to a file or log aggregator; for a portfolio
# project, stdout is fine (Render captures this in its log viewer).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

REQUIRED_FEATURES = [f"V{i}" for i in range(1, 29)]
MAX_BATCH_SIZE = 500  # prevents someone from sending 1M rows and hanging the server

app = FastAPI(
    title="AI Fraud Detection API",
    version="1.1.0",
    description="Real-time and batch fraud risk scoring with SHAP-backed model.",
)

# Allow the Streamlit frontend (or any frontend) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading — never crash the whole app if this fails. Instead, keep
# model=None and return a clear 503 on every endpoint that needs it.
# ---------------------------------------------------------------------------
model = None
scaler = None
model_load_error = None

try:
    init_db()
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    model_load_error = str(e)
    logger.error(f"Model files not found: {e}")
except Exception as e:
    model_load_error = str(e)
    logger.error(f"Unexpected error loading model: {e}")


def get_feature_names(model) -> List[str]:
    """Works with both XGBoost (get_booster) and scikit-learn models (feature_names_in_)."""
    if hasattr(model, "get_booster"):
        return list(model.get_booster().feature_names)
    return list(model.feature_names_in_)


def score_transaction(amount: float, time_val: float, features: dict):
    """
    Shared scoring logic. Raises HTTPException with a clear message on any
    failure instead of letting a raw exception (and 500 error) reach the client.
    """
    try:
        row = features.copy()
        row["scaled_amount"] = scaler.transform([[amount]])[0][0]
        row["scaled_time"] = scaler.transform([[time_val]])[0][0]

        df = pd.DataFrame([row])

        expected_cols = get_feature_names(model)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature(s): {missing}",
            )

        df = df[expected_cols]

        fraud_probability = float(model.predict_proba(df)[0][1])
        is_fraud = fraud_probability > 0.5
        risk_level = (
            "High" if fraud_probability > 0.7
            else "Medium" if fraud_probability > 0.3
            else "Low"
        )
        return fraud_probability, is_fraud, risk_level

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class Transaction(BaseModel):
    Amount: float = Field(..., ge=0, description="Transaction amount, must be non-negative")
    Time: float = Field(..., ge=0, description="Seconds since first transaction in the dataset")
    features: dict = Field(..., description="V1 through V28 PCA feature values")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        missing = [f for f in REQUIRED_FEATURES if f not in v]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        non_numeric = [k for k, val in v.items() if not isinstance(val, (int, float))]
        if non_numeric:
            raise ValueError(f"Non-numeric feature values for: {non_numeric}")
        return v


class BatchTransaction(BaseModel):
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)
    features: dict


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request with method, path, status code, and latency."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running", "model_loaded": model is not None}


@app.get("/health")
def health():
    """
    Real health check for uptime monitoring / load balancers.
    Distinguishes between 'server is up' and 'server is actually functional'.
    """
    if model is None or scaler is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": model_load_error,
        }
    return {"status": "ok", "model_loaded": True, "model_type": type(model).__name__}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The server may be misconfigured — check /health.",
        )

    fraud_probability, is_fraud, risk_level = score_transaction(
        transaction.Amount, transaction.Time, transaction.features
    )

    try:
        save_prediction(
            amount=transaction.Amount,
            time_val=transaction.Time,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            risk_level=risk_level,
        )
    except Exception as e:
        # Don't fail the whole request just because logging to DB failed —
        # the prediction itself is still valid and should be returned.
        logger.error(f"Failed to save prediction to database: {e}")

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


@app.post("/predict/batch")
def predict_batch(transactions: List[BatchTransaction]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    if len(transactions) == 0:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")

    if len(transactions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(transactions)} rows (max {MAX_BATCH_SIZE}). Split into smaller batches.",
        )

    results = []
    errors = []

    for i, transaction in enumerate(transactions):
        try:
            fraud_probability, is_fraud, risk_level = score_transaction(
                transaction.Amount, transaction.Time, transaction.features
            )
            try:
                save_prediction(
                    amount=transaction.Amount,
                    time_val=transaction.Time,
                    fraud_probability=fraud_probability,
                    is_fraud=is_fraud,
                    risk_level=risk_level,
                )
            except Exception as e:
                logger.error(f"Row {i}: failed to save to database: {e}")

            results.append({
                "row": i,
                "fraud_probability": round(fraud_probability, 4),
                "is_fraud": is_fraud,
                "risk_level": risk_level,
            })
        except HTTPException as e:
            # A single bad row shouldn't kill the whole batch — record the
            # error for that row and keep processing the rest.
            errors.append({"row": i, "error": e.detail})
            results.append({"row": i, "fraud_probability": None, "is_fraud": None, "risk_level": "Error"})

    return {
        "results": results,
        "total": len(results),
        "flagged": sum(1 for r in results if r["is_fraud"]),
        "errors": errors,
    }


@app.get("/history")
def history(limit: int = 100):
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000.")
    try:
        return get_history(limit)
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve prediction history.")