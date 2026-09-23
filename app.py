import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Investigation Agent",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #ffffff;
        font-weight: 800;
        padding-bottom: 0px;
    }
    h2, h3 {
        color: #e6e6e6;
        font-weight: 600;
    }
    [data-testid="stMetric"] {
        background-color: #1a1d27;
        border: 1px solid #2d3140;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 10px;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI Transaction Risk & Fraud Investigation Agent")
st.write("This dashboard uses simulated transaction data for demonstration purposes only.")

# --- Load data: uploaded file, or fall back to sample data ---
st.subheader("Load Transaction Data")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

use_sample = st.button("Or try it with sample data")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Loaded your uploaded file.")
elif use_sample:
    df = pd.read_csv("sample_transactions.csv")
    st.info("Showing sample data.")
else:
    st.info("Upload a CSV file above to get started, or click the button to try sample data.")
    st.stop()


# --- Rule-based flagging logic ---
def evaluate_transaction(row):
    reasons = []
    score = 0

    if row["amount"] > 500:
        reasons.append("Large amount (> $500)")
        score += 40

    if row["amount"] > 2000:
        reasons.append("Very large amount (> $2000)")
        score += 30

    if row["hour"] < 6 or row["hour"] > 23:
        reasons.append("Unusual hour")
        score += 20

    if "unknown" in row["merchant"].lower():
        reasons.append("Vague/unknown merchant")
        score += 25

    return pd.Series([reasons, score])


def score_to_level(score):
    if score >= 50:
        return "High"
    elif score >= 20:
        return "Medium"
    else:
        return "Low"


def highlight_risk(val):
    if val == "High":
        return "background-color: #3b0d0d; color: #ff6b6b; font-weight: 600;"
    elif val == "Medium":
        return "background-color: #3b2e0d; color: #ffd166; font-weight: 600;"
    elif val == "Low":
        return "background-color: #0d3b1a; color: #6bff8f; font-weight: 600;"
    return ""


def highlight_fraud(val):
    if val == 1:
        return "background-color: #3b0d0d; color: #ff6b6b; font-weight: 600;"
    return ""


# Check if this data matches our rule-based engine's expected format
rule_based_columns = {"amount", "merchant", "hour"}
has_rule_columns = rule_based_columns.issubset(df.columns)

if has_rule_columns:
    df[["flag_reasons", "risk_score"]] = df.apply(evaluate_transaction, axis=1)
    df["risk_level"] = df["risk_score"].apply(score_to_level)

    st.subheader("Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("High Risk", (df["risk_level"] == "High").sum())
    col3.metric("Medium Risk", (df["risk_level"] == "Medium").sum())
    col4.metric("Low Risk", (df["risk_level"] == "Low").sum())

    st.subheader("Risk Level Breakdown")
    risk_counts = df["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("All Transactions")
    styled_df = df.style.map(highlight_risk, subset=["risk_level"])
    st.dataframe(styled_df)

    # --- Analyst review section ---
    st.subheader("Investigate Flagged Transactions")

    if "review_status" not in st.session_state:
        st.session_state.review_status = {
            txn_id: "Not Reviewed" for txn_id in df["transaction_id"]
        }

    flagged_df = df[df["risk_level"] != "Low"]

    for _, row in flagged_df.iterrows():
        with st.container(border=True):
            st.write(f"**Transaction #{row['transaction_id']}** — {row['merchant']}")
            st.write(f"Amount: ${row['amount']:.2f} | Hour: {row['hour']} | Risk: {row['risk_level']} (score: {row['risk_score']})")
            st.write(f"Reasons: {', '.join(row['flag_reasons'])}")

            selected_status = st.selectbox(
                "Review status",
                options=["Not Reviewed", "Confirmed Fraud", "False Positive"],
                index=["Not Reviewed", "Confirmed Fraud", "False Positive"].index(
                    st.session_state.review_status[row["transaction_id"]]
                ),
                key=f"status_{row['transaction_id']}"
            )

            st.session_state.review_status[row["transaction_id"]] = selected_status

    # --- Export report ---
    st.subheader("Export Report")

    export_df = flagged_df.copy()
    export_df["review_status"] = export_df["transaction_id"].map(st.session_state.review_status)
    csv_data = export_df.to_csv(index=False)

    st.download_button(
        label="Download Flagged Transactions Report (CSV)",
        data=csv_data,
        file_name="flagged_transactions_report.csv",
        mime="text/csv"
    )

else:
    st.warning("This file doesn't match the rule-based engine's expected format (amount, merchant, hour columns). Rule-based detection skipped for this file — see ML section below instead.")


# --- ML-based fraud detection (real dataset format only) ---
st.subheader("Machine Learning Fraud Detection")

expected_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
has_ml_columns = all(col in df.columns for col in expected_columns)

if has_ml_columns:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")

    X = df.drop(columns=["Class"], errors="ignore")
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    fraud_probabilities = model.predict_proba(X_scaled)[:, 1]

    df["ml_prediction"] = predictions
    df["ml_fraud_probability"] = fraud_probabilities

    ml_flagged = df[df["ml_prediction"] == 1]

    st.success(f"ML model ran on {len(df)} transactions.")
    col1, col2 = st.columns(2)
    col1.metric("Flagged by ML Model", len(ml_flagged))
    col2.metric("Total Transactions", len(df))

    st.write("Transactions flagged by the ML model:")
    styled_ml = ml_flagged.style.map(highlight_fraud, subset=["ml_prediction"])
    st.dataframe(styled_ml)

else:
    st.info("Upload a file with the real dataset's columns (Time, V1–V28, Amount) to run ML-based detection. The current data uses a different format, so only rule-based detection applies above.")
