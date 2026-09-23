import streamlit as st
import pandas as pd
import joblib

st.title("AI Transaction Risk & Fraud Investigation Agent")
st.write("This dashboard uses simulated transaction data for demonstration purposes only.")

# --- Load data: uploaded file, or fall back to sample data ---
st.subheader("Load Transaction Data")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Let the analyst optionally load sample data instead of uploading
use_sample = st.button("Or try it with sample data")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Loaded your uploaded file.")
elif use_sample:
    df = pd.read_csv("sample_transactions.csv")
    st.info("Showing sample data.")
else:
    st.info("Upload a CSV file above to get started, or click the button to try sample data.")
    st.stop()  # Halts the app here — nothing below this runs until data is loaded


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


# Check if this data matches our rule-based engine's expected format
rule_based_columns = {"amount", "merchant", "hour"}
has_rule_columns = rule_based_columns.issubset(df.columns)

# --- Everything rule-based lives inside this block now ---
if has_rule_columns:
    df[["flag_reasons", "risk_score"]] = df.apply(evaluate_transaction, axis=1)
    df["risk_level"] = df["risk_score"].apply(score_to_level)

    # --- Dashboard summary section ---
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
    st.dataframe(df)

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
    st.dataframe(ml_flagged)

else:
    st.info("Upload a file with the real dataset's columns (Time, V1–V28, Amount) to run ML-based detection. The current data uses a different format, so only rule-based detection applies above.")
    