# AI-Based Fraud Detection System

## Setup
1. `venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Download creditcard.csv from Kaggle and place in data\raw\
4. `python src\preprocessing.py`
5. `python src\train_model.py`
6. `python src\evaluate.py`
7. `uvicorn api.main:app --reload`