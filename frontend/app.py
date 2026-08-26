"""
app.py — Streamlit Dashboard
------------------------------
A simple UI for the fraud detection system. Talks to the FastAPI
backend deployed on Render.

Run:  streamlit run frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://ai-fraud-detection-system-3ya6.onrender.com"

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ AI-Based Fraud Detection System")
st.caption("Real-time transaction fraud risk scoring powered by XGBoost + SHAP")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload (CSV)", "📜 History", "📊 Analytics"])

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
        with st.spinner("Checking transaction... (may take up to 50s if the API was idle)"):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
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
                st.error("Could not reach the API. It may be starting up — try again in a moment.")
            except requests.exceptions.ReadTimeout:
                st.error("Request timed out. The API might be waking up from sleep — please try again.")

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
            payload = [
                {
                    "Amount": row["Amount"],
                    "Time": row["Time"],
                    "features": {f"V{j}": row[f"V{j}"] for j in range(1, 29)},
                }
                for _, row in df.iterrows()
            ]

            with st.spinner(f"Scoring {len(payload)} transactions... (may take up to 60s if the API was idle)"):
                try:
                    r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=120)
                    response_data = r.json()

                    results_df = pd.DataFrame(response_data["results"])
                    final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

                    st.success(f"Done! {response_data['flagged']} of {response_data['total']} transactions flagged as fraud.")
                    st.dataframe(final)
                    st.metric("Flagged as Fraud", response_data["flagged"])

                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the API. It may be starting up — try again in a moment.")
                except requests.exceptions.ReadTimeout:
                    st.error("Request timed out. Try again, or reduce the batch size.")

# ---------------- TAB 3: History ----------------
with tab3:
    st.subheader("Prediction History")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/history", timeout=60)
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
        st.error("Could not reach the API.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. The API might be waking up from sleep — please try again.")

# ---------------- TAB 4: Analytics ----------------
with tab4:
    st.subheader("Fraud Analytics Dashboard")

    try:
        response = requests.get(f"{API_URL}/history", timeout=60)
        history = response.json()

        if not history:
            st.info("No data yet — make some predictions first (Single Prediction or Batch Upload tabs).")
        else:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["fraud_probability_pct"] = df["fraud_probability"] * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", len(df))
            col2.metric("Flagged as Fraud", int(df["is_fraud"].sum()))
            fraud_rate = (df["is_fraud"].sum() / len(df)) * 100
            col3.metric("Fraud Rate", f"{fraud_rate:.1f}%")
            col4.metric("Avg Amount", f"${df['amount'].mean():.2f}")

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Risk Level Distribution**")
                risk_counts = df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                fig1 = px.pie(
                    risk_counts, names="Risk Level", values="Count",
                    color="Risk Level",
                    color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col_b:
                st.markdown("**Fraud Probability Distribution**")
                fig2 = px.histogram(
                    df, x="fraud_probability_pct", nbins=20,
                    labels={"fraud_probability_pct": "Fraud Probability (%)"},
                    color_discrete_sequence=["#3498db"],
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Predictions Over Time**")
            df_sorted = df.sort_values("timestamp")
            df_sorted["cumulative_fraud"] = df_sorted["is_fraud"].cumsum()
            fig3 = px.line(
                df_sorted, x="timestamp", y="cumulative_fraud",
                labels={"cumulative_fraud": "Cumulative Fraud Count", "timestamp": "Time"},
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("**Transaction Amount vs Fraud Probability**")
            fig4 = px.scatter(
                df, x="amount", y="fraud_probability_pct",
                color="risk_level",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                labels={"amount": "Amount ($)", "fraud_probability_pct": "Fraud Probability (%)"},
            )
            st.plotly_chart(fig4, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. The API might be waking up from sleep — please try again.")