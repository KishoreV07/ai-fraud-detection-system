"""
app.py — Streamlit Dashboard (marketplace-inspired theme)
-----------------------------------------------------------
Fraud detection UI styled like a modern e-commerce trust/safety
dashboard (Amazon/Flipkart-inspired badges, cards, gauges) while
talking to the same FastAPI backend on Render.

Run:  streamlit run frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from auth import check_login, logout_button

API_URL = "https://ai-fraud-detection-system-3ya6.onrender.com"

st.set_page_config(
    page_title="Fraud Shield | AI Trust & Safety Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME
# Palette: deep navy/slate base (fintech) + amber/gold + electric blue accents
# (marketplace "verified" / "best seller" energy). Card-first layout,
# badges, hover lift, animated pulse on live status.
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif; letter-spacing: -0.3px; }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #101a30 0%, #0B1120 45%) fixed;
    }

    /* ---------- Top banner / hero ---------- */
    .hero-banner {
        background: linear-gradient(120deg, #0F1B33 0%, #132244 55%, #0B1120 100%);
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, #FFB02E33 0%, transparent 70%);
    }
    .brand-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        background: linear-gradient(90deg, #FFD166 0%, #F8FAFC 60%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-sub { color: #94A3B8; font-size: 0.92rem; margin-top: 2px; }

    /* pulsing live-status dot */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700;
        background-color: #10B98118; color: #34D399;
        border: 1px solid #10B98155;
        float: right;
    }
    .status-pill.warn { background-color: #F59E0B18; color: #FBBF24; border-color: #F59E0B55; }
    .status-pill.bad  { background-color: #EF444418; color: #F87171; border-color: #EF444455; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor;
           box-shadow: 0 0 0 currentColor; animation: pulse 1.6s infinite; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
        70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    /* ---------- Trust badges (Amazon/Flipkart-style) ---------- */
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 6px;
        font-size: 0.8rem; font-weight: 700; margin-bottom: 8px;
        letter-spacing: 0.2px;
    }
    .badge-safe   { background: linear-gradient(90deg,#10B981,#059669); color: #05261c; }
    .badge-review { background: linear-gradient(90deg,#F59E0B,#D97706); color: #2b1a02; }
    .badge-risk   { background: linear-gradient(90deg,#EF4444,#B91C1C); color: #fff; }

    /* ---------- Cards ---------- */
    .fs-card {
        background: linear-gradient(180deg,#131C31 0%, #101828 100%);
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .fs-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px -8px rgba(14,165,233,0.25);
        border-color: #0EA5E955;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg,#131C31 0%, #101828 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -8px rgba(14,165,233,0.3);
    }
    div[data-testid="stMetricLabel"] { color: #94A3B8; }

    /* ---------- Tabs, marketplace-category style ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1E293B; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #17233d, #101828);
        border-bottom: 3px solid #FFB02E !important;
        color: #FFD166 !important;
    }

    /* ---------- Buttons: gold CTA like "Buy Now" ---------- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FFB02E, #F59E0B);
        color: #1a1200;
        border: none;
        font-weight: 700;
        border-radius: 10px;
        box-shadow: 0 6px 16px -6px rgba(245,158,11,0.55);
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 10px 22px -6px rgba(245,158,11,0.65);
        background: linear-gradient(90deg, #FFC65A, #FBBF24);
    }

    .stAlert { border-radius: 10px; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1728 0%, #0B1120 100%);
        border-right: 1px solid #1E293B;
    }

    /* ---------- Gauge label ---------- */
    .gauge-caption { text-align:center; color:#94A3B8; font-size:0.85rem; margin-top:-6px; }

    /* ---------- Alert / flagged item card ---------- */
    .flag-card {
        background: linear-gradient(180deg,#1a1220,#150c14);
        border: 1px solid #7f1d1d66;
        border-left: 4px solid #EF4444;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }

    /* fade-in for freshly rendered blocks */
    .fadein { animation: fadeIn .4s ease; }
    @keyframes fadeIn { from {opacity:0; transform: translateY(4px);} to {opacity:1; transform:none;} }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if not check_login():
    st.stop()

# ---------------------------------------------------------------------------
# Session-level "activity" counters — real, derived from this session's
# actual actions (not fabricated), shown sidebar-cart-style.
# ---------------------------------------------------------------------------
if "checks_this_session" not in st.session_state:
    st.session_state.checks_this_session = 0
if "flags_this_session" not in st.session_state:
    st.session_state.flags_this_session = 0


def api_status():
    """Returns (label, css_class) — never raises."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200 and r.json().get("model_loaded"):
            return "API Online", ""
        return "API Degraded", "warn"
    except Exception:
        return "API Waking Up", "bad"


def risk_badge_html(risk_level: str) -> str:
    if risk_level == "High":
        return '<span class="badge badge-risk">🚫 High Risk — Review Recommended</span>'
    elif risk_level == "Medium":
        return '<span class="badge badge-review">⚠️ Needs Manual Review</span>'
    else:
        return '<span class="badge badge-safe">✅ Verified Safe</span>'


def risk_color(risk_level: str) -> str:
    return {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}.get(risk_level, "#0EA5E9")


def render_gauge(prob_pct: float, risk_level: str):
    """Circular gauge for fraud probability, marketplace-'rating dial' style."""
    color = risk_color(risk_level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 34, "family": "Poppins"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8", "tickfont": {"color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#10B98122"},
                {"range": [30, 70], "color": "#F59E0B22"},
                {"range": [70, 100], "color": "#EF444422"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=0),
        height=220,
    )
    st.plotly_chart(fig, use_container_width=True)


logout_button()

# ---------------- HERO / HEADER ----------------
status_label, status_class = api_status()
st.markdown(f"""
<div class="hero-banner fadein">
    <span class="status-pill {status_class}"><span class="dot"></span>{status_label}</span>
    <div class="brand-title">🛡️ Fraud Shield</div>
    <div class="brand-sub">AI-powered transaction trust & safety · XGBoost / Random Forest · real-time & batch scoring</div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR: session activity ----------------
with st.sidebar:
    st.markdown("### 🧾 This Session")
    sc1, sc2 = st.columns(2)
    sc1.metric("Checks Run", st.session_state.checks_this_session)
    sc2.metric("Flags Raised", st.session_state.flags_this_session)
    st.divider()
    st.markdown("### 📖 Legend")
    st.markdown('<span class="badge badge-safe">✅ Verified Safe</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-review">⚠️ Needs Review</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-risk">🚫 High Risk</span>', unsafe_allow_html=True)
    st.divider()
    st.caption("Risk bands: 0–30% Low · 30–70% Medium · 70–100% High")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Check Transaction", "📁  Batch Analysis", "📜  History", "📊  Analytics", "🚨  Alerts"
])

# ---------------- TAB 1: Single Transaction ----------------
with tab1:
    st.subheader("Real-time transaction check")
    st.caption("Like a marketplace buyer-protection scan — score a single transaction instantly.")

    left, right = st.columns([1.1, 1])

    with left:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            with col2:
                time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)

            st.caption("V1–V28 are anonymized PCA features from the dataset. Defaults represent a typical legitimate transaction.")

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

            analyze_clicked = st.button("🔎 Analyze Transaction", type="primary", use_container_width=True)

    with right:
        result_slot = st.container(border=True)
        with result_slot:
            st.markdown("**Scan Result**")
            gauge_placeholder = st.empty()
            badge_placeholder = st.empty()
            if not analyze_clicked:
                gauge_placeholder.info("Run a scan to see the live risk dial here.")

    if analyze_clicked:
        payload = {"Amount": amount, "Time": time, "features": features}
        with st.spinner("Analyzing transaction... (first request may take up to 50s if the API was idle)"):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)

                if response.status_code == 422:
                    result_slot.error(f"Invalid input: {response.json().get('detail')}")
                elif response.status_code == 503:
                    result_slot.error("The model isn't loaded on the server right now. Please try again shortly.")
                elif response.status_code != 200:
                    result_slot.error(f"Unexpected error (HTTP {response.status_code}). Please try again.")
                else:
                    result = response.json()
                    prob = result["fraud_probability"]
                    risk = result["risk_level"]

                    st.session_state.checks_this_session += 1
                    if result["is_fraud"]:
                        st.session_state.flags_this_session += 1

                    with gauge_placeholder.container():
                        render_gauge(prob * 100, risk)
                    badge_placeholder.markdown(
                        f'<div class="fadein" style="text-align:center;">{risk_badge_html(risk)}</div>',
                        unsafe_allow_html=True,
                    )

                    st.divider()
                    if risk == "High":
                        st.error("⚠️ **HIGH RISK** — likely fraudulent transaction. Recommend manual review before approval.")
                    elif risk == "Medium":
                        st.warning("⚠️ **MEDIUM RISK** — flagged for review, not auto-blocked.")
                    else:
                        st.success("✅ **LOW RISK** — transaction appears legitimate.")
                        st.toast("Transaction cleared ✅", icon="🛡️")

            except requests.exceptions.ConnectionError:
                result_slot.error("Could not reach the API. It may be starting up — please try again in a moment.")
            except requests.exceptions.ReadTimeout:
                result_slot.error("Request timed out. The API might be waking up from sleep — please try again.")
            except Exception as e:
                result_slot.error(f"Unexpected error: {e}")

# ---------------- TAB 2: Batch CSV Upload ----------------
with tab2:
    st.subheader("Batch transaction analysis")
    st.caption("Upload a CSV with columns: Amount, Time, V1, V2, ... V28 (max 500 rows per batch)")

    with st.container(border=True):
        uploaded_file = st.file_uploader("📤 Drop your CSV here, or browse", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file as CSV: {e}")
            df = None

        if df is not None:
            required_cols = {"Amount", "Time"} | {f"V{i}" for i in range(1, 29)}
            missing_cols = required_cols - set(df.columns)

            if missing_cols:
                st.error(f"CSV is missing required column(s): {sorted(missing_cols)}")
            elif len(df) == 0:
                st.warning("The uploaded file has no rows.")
            elif len(df) > 500:
                st.error(f"File has {len(df)} rows — please split into batches of 500 or fewer.")
            else:
                st.success(f"Loaded {len(df)} transactions — ready to scan.")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("🚀 Run Batch Analysis", type="primary"):
                    payload = [
                        {
                            "Amount": row["Amount"],
                            "Time": row["Time"],
                            "features": {f"V{j}": row[f"V{j}"] for j in range(1, 29)},
                        }
                        for _, row in df.iterrows()
                    ]

                    progress = st.progress(0, text="Sending batch to the model...")
                    with st.spinner(f"Scoring {len(payload)} transactions..."):
                        try:
                            progress.progress(30, text="Scoring in progress...")
                            r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=120)
                            progress.progress(80, text="Assembling results...")

                            if r.status_code != 200:
                                st.error(f"Batch request failed (HTTP {r.status_code}): {r.text[:200]}")
                            else:
                                response_data = r.json()
                                results_df = pd.DataFrame(response_data["results"])
                                final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

                                progress.progress(100, text="Done")
                                progress.empty()

                                if response_data.get("errors"):
                                    st.warning(f"{len(response_data['errors'])} row(s) failed to score — see 'Error' rows below.")

                                flagged = response_data["flagged"]
                                total = response_data["total"]
                                st.session_state.checks_this_session += total
                                st.session_state.flags_this_session += flagged

                                m1, m2, m3 = st.columns(3)
                                m1.metric("Total Scanned", total)
                                m2.metric("Flagged as Fraud", flagged)
                                m3.metric("Clean Rate", f"{(1 - flagged / total) * 100:.1f}%" if total else "—")

                                st.success(f"Done! {flagged} of {total} transactions flagged as fraud.")

                                def _highlight_risk(row):
                                    r = row.get("risk_level")
                                    color = risk_color(r) if r in ("Low", "Medium", "High") else "#334155"
                                    return [f"background-color: {color}22" for _ in row]

                                st.dataframe(
                                    final.style.apply(_highlight_risk, axis=1),
                                    use_container_width=True,
                                )

                                csv_out = final.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "📥 Download Scored Batch (CSV)",
                                    data=csv_out,
                                    file_name="fraud_batch_results.csv",
                                    mime="text/csv",
                                )

                        except requests.exceptions.ConnectionError:
                            st.error("Could not reach the API. It may be starting up — try again in a moment.")
                        except requests.exceptions.ReadTimeout:
                            st.error("Request timed out. Try again, or reduce the batch size.")

# ---------------- TAB 3: History ----------------
with tab3:
    st.subheader("Prediction history")

    c_ref, c_gap = st.columns([1, 5])
    if c_ref.button("🔄 Refresh"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/history", timeout=60)

        if response.status_code != 200:
            st.error(f"Could not load history (HTTP {response.status_code}).")
        else:
            history = response.json()

            if not history:
                st.info("No predictions yet — try the Check Transaction tab first.")
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

                def _highlight_risk_hist(row):
                    r = row.get("risk_level")
                    color = risk_color(r) if r in ("Low", "Medium", "High") else "#334155"
                    return [f"background-color: {color}1a" for _ in row]

                st.dataframe(
                    hist_df.style.apply(_highlight_risk_hist, axis=1),
                    use_container_width=True,
                )

                csv_data = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download History as CSV",
                    data=csv_data,
                    file_name="fraud_prediction_history.csv",
                    mime="text/csv",
                )

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. The API might be waking up from sleep — please try again.")

# ---------------- TAB 4: Analytics ----------------
with tab4:
    st.subheader("Fraud analytics")

    try:
        response = requests.get(f"{API_URL}/history", timeout=60)

        if response.status_code != 200:
            st.error(f"Could not load analytics data (HTTP {response.status_code}).")
        else:
            history = response.json()

            if not history:
                st.info("No data yet — make some predictions first.")
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
                        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                        hole=0.55,
                    )
                    fig1.update_traces(textfont_color="#F8FAFC")
                    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        legend=dict(font=dict(color="#94A3B8")))
                    st.plotly_chart(fig1, use_container_width=True)

                with col_b:
                    st.markdown("**Fraud Probability Distribution**")
                    fig2 = px.histogram(
                        df, x="fraud_probability_pct", nbins=20,
                        labels={"fraud_probability_pct": "Fraud Probability (%)"},
                        color_discrete_sequence=["#0EA5E9"],
                    )
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#94A3B8"))
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("**Predictions Over Time**")
                df_sorted = df.sort_values("timestamp")
                df_sorted["cumulative_fraud"] = df_sorted["is_fraud"].cumsum()
                fig3 = px.area(
                    df_sorted, x="timestamp", y="cumulative_fraud",
                    labels={"cumulative_fraud": "Cumulative Fraud Count", "timestamp": "Time"},
                )
                fig3.update_traces(line_color="#0EA5E9", fillcolor="rgba(14,165,233,0.15)")
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#94A3B8"))
                st.plotly_chart(fig3, use_container_width=True)

                st.markdown("**Transaction Amount vs Fraud Probability**")
                fig4 = px.scatter(
                    df, x="amount", y="fraud_probability_pct",
                    color="risk_level",
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                    labels={"amount": "Amount ($)", "fraud_probability_pct": "Fraud Probability (%)"},
                )
                fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#94A3B8"))
                st.plotly_chart(fig4, use_container_width=True)

                st.markdown("**Avg. Amount by Risk Level**")
                avg_amt = df.groupby("risk_level")["amount"].mean().reindex(["Low", "Medium", "High"]).dropna().reset_index()
                fig5 = px.bar(
                    avg_amt, x="risk_level", y="amount", color="risk_level",
                    color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
                    labels={"amount": "Avg Amount ($)", "risk_level": "Risk Level"},
                )
                fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#94A3B8"), showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. The API might be waking up from sleep — please try again.")

# ---------------- TAB 5: Alerts ----------------
with tab5:
    st.subheader("Fraud alerts")
    st.caption("Transactions flagged as fraud by the model, most recent first — reviewed like flagged marketplace orders.")

    try:
        response = requests.get(f"{API_URL}/history", timeout=60)

        if response.status_code != 200:
            st.error(f"Could not load alerts (HTTP {response.status_code}).")
        else:
            history = response.json()

            if not history:
                st.info("No predictions yet.")
            else:
                df = pd.DataFrame(history)
                alerts = df[df["is_fraud"] == True].copy()

                if alerts.empty:
                    st.success("✅ No fraud-flagged transactions detected. All clear!")
                else:
                    alerts["fraud_probability"] = (alerts["fraud_probability"] * 100).round(2)
                    alerts = alerts.sort_values("timestamp", ascending=False)

                    st.error(f"⚠️ {len(alerts)} transaction(s) flagged as fraud — review recommended")

                    alerts_csv = alerts.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Alerts Report (CSV)",
                        data=alerts_csv,
                        file_name="fraud_alerts_report.csv",
                        mime="text/csv",
                    )

                    for _, row in alerts.iterrows():
                        prob = row["fraud_probability"]
                        risk = row.get("risk_level", "High")
                        st.markdown(f"""
                        <div class="flag-card fadein">
                            {risk_badge_html(risk)}
                            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-top:4px;">
                                <div><span style="color:#94A3B8;">Amount</span><br>
                                     <b style="font-size:1.1rem;">${row['amount']:.2f}</b></div>
                                <div><span style="color:#94A3B8;">Fraud Probability</span><br>
                                     <b style="font-size:1.1rem; color:{risk_color(risk)};">{prob:.2f}%</b></div>
                                <div><span style="color:#94A3B8;">Flagged At</span><br>
                                     <b>{str(row['timestamp'])[:19]}</b></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. The API might be waking up from sleep — please try again.")