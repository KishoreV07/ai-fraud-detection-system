"""
app.py — Streamlit Dashboard
------------------------------
A simple UI for the fraud detection system. Talks to the FastAPI
backend running at http://127.0.0.1:8000

Run:  streamlit run frontend/app.py
(Make sure the API is running in a separate terminal first!)
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ AI-Based Fraud Detection System")
st.caption("Real-time transaction fraud risk scoring powered by XGBoost + SHAP")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload (CSV)", "📜 History"])

# ---------------- TAB 1: Single Transaction ----------------
with tab1:
    st.subheader("Check a single transaction")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    with col2:
        time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)

    st.caption("V1–V28 are anonymized PCA features from the dataset. Defaults are a sample legitimate transaction.")

    default_v = {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
        "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
        "V26": -0.19, "V27": 0.13, "V28": -0.02,
    }

    with st.expander("Advanced: edit V1–V28 features"):
        features = {}
        cols = st.columns(4)
        for i, (key, val) in enumerate(default_v.items()):
            features[key] = cols[i % 4].number_input(key, value=val, key=key)

    if st.button("🔎 Check for Fraud", type="primary"):
        payload = {"Amount": amount, "Time": time, "features": features}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            result = response.json()

            prob = result["fraud_probability"]
            risk = result["risk_level"]

            st.metric("Fraud Probability", f"{prob * 100:.2f}%")

            if risk == "High":
                st.error("⚠️ HIGH RISK — likely fraudulent transaction")
            elif risk == "Medium":
                st.warning("⚠️ MEDIUM RISK — needs review")
            else:
                st.success("✅ LOW RISK — looks legitimate")

        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Make sure it's running: `uvicorn api.main:app --reload`")

# ---------------- TAB 2: Batch CSV Upload ----------------
with tab2:
    st.subheader("Upload a CSV of transactions")
    st.caption("CSV must have columns: Amount, Time, V1, V2, ... V28")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} transactions")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            results = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                features = {f"V{j}": row[f"V{j}"] for j in range(1, 29)}
                payload = {"Amount": row["Amount"], "Time": row["Time"], "features": features}
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    results.append(r.json())
                except Exception:
                    results.append({"fraud_probability": None, "is_fraud": None, "risk_level": "Error"})
                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

            st.success("Done!")
            st.dataframe(final)

            fraud_count = final["is_fraud"].sum()
            st.metric("Flagged as Fraud", int(fraud_count))

# ---------------- TAB 3: History ----------------
with tab3:
    st.subheader("Prediction History")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/history", timeout=5)
        history = response.json()

        if not history:
            st.info("No predictions yet — try the Single Prediction tab first.")
        else:
            hist_df = pd.DataFrame(history)
            hist_df["fraud_probability"] = (hist_df["fraud_probability"] * 100).round(2)
            hist_df = hist_df.rename(columns={
                "fraud_probability": "fraud_probability (%)",
                "time_val": "time",
            })

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", len(hist_df))
            col2.metric("Flagged as Fraud", int(hist_df["is_fraud"].sum()))
            col3.metric("Avg Fraud Probability", f"{hist_df['fraud_probability (%)'].mean():.2f}%")

            st.dataframe(hist_df, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Make sure it's running: `uvicorn api.main:app --reload`")