# 🛡️ AI-Based Fraud Detection System

Real-time transaction fraud detection using machine learning, explainable AI, and a full-stack deployment — built as an end-to-end system, not just a notebook.

## Highlights
- **0.97 ROC-AUC** on the Kaggle Credit Card Fraud dataset (284K+ transactions)
- **XGBoost** classifier with SMOTE-balanced training
- **SHAP** explainability — see exactly why a transaction was flagged
- **FastAPI** REST backend with SQLite prediction logging
- **Streamlit** dashboard — single prediction, batch CSV upload, and history view
- **Dockerized** for deployment

## Tech Stack
Python · scikit-learn · XGBoost · SHAP · FastAPI · Streamlit · SQLite · Docker

## Architecture
```
Data (Kaggle) → Preprocessing (SMOTE) → XGBoost Model → FastAPI → Streamlit Dashboard
                                                              ↓
                                                    SQLite (prediction history)
```

## Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.9676 |
| Precision (fraud class) | 0.71 |
| Recall (fraud class) | 0.81 |
