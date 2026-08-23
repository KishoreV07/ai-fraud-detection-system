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

## Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src\preprocessing.py
python src\train_model.py
python src\evaluate.py

uvicorn api.main:app --reload
# in a new terminal:
streamlit run frontend\app.py
```

## Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.9676 |
| Precision (fraud class) | 0.71 |
| Recall (fraud class) | 0.81 |
