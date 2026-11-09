import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib

RULE_SAMPLE_FILE = "sample_transactions.csv"
ML_SAMPLE_FILE = "demo_creditcard_sample.csv"
RECOMMENDED_THRESHOLD = 0.95
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

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
        letter-spacing: -0.5px;
        padding-bottom: 0px;
    }
    h2, h3 {
        color: #e6e6e6;
        font-weight: 600;
    }
    p, .stMarkdown {
        color: #c9ccd1;
    }
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2d3140;
    }
    [data-testid="stMetric"] {
        background-color: #1a1d27;
        border: 1px solid #2d3140;
        border-radius: 12px;
        padding: 18px;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2d3140;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 10px;
        background-color: #131722;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    .stButton button, .stDownloadButton button {
        background-color: #1a1d27;
        color: #ffffff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #3b82f6;
        border: 1px solid #3b82f6;
    }
    [data-testid="stTabs"] button {
        font-weight: 600;
    }
    [data-testid="stContainer"] {
        border-radius: 10px;
    }
    hr {
        border-color: #2d3140;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ Fraud Agent")
    st.markdown("Simulated transaction risk analysis for demonstration purposes.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("- Upload a CSV, or try the sample data\n- Rule-based engine flags known patterns\n- ML model scores real transaction data\n- Review and export findings")
    st.markdown("---")
    st.markdown("**Test data**")
    st.caption("Download these files, then upload them above to try the app yourself.")
    for path, label in [
        (RULE_SAMPLE_FILE, "Download rule-based sample (CSV)"),
        (ML_SAMPLE_FILE, "Download ML demo sample (CSV)"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(label, f.read(), file_name=path, mime="text/csv", key=f"dl_{path}")
    st.caption(
        "ML demo sample: 2,000 transactions drawn from the held-out test split of the Kaggle "
        "Credit Card Fraud Detection dataset (ULB Machine Learning Group / Worldline). "
        "Fraud cases are over-represented so the demo is easy to see."
    )
    st.markdown(f"[Get the full Kaggle dataset]({KAGGLE_URL})")
    st.markdown("---")
    st.caption("Built with Streamlit, pandas, and scikit-learn.")

st.title("AI Transaction Risk & Fraud Investigation Agent")
st.write("This dashboard uses simulated transaction data for demonstration purposes only.")

# --- Load data: uploaded file, or one of the built-in samples ---
st.subheader("Load Transaction Data")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

btn_col1, btn_col2 = st.columns(2)
if btn_col1.button("Try rule-based sample data"):
    st.session_state.sample_choice = "rule"
if btn_col2.button("Try ML demo data"):
    st.session_state.sample_choice = "ml"

sample_choice = st.session_state.get("sample_choice")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Loaded your uploaded file. (Remove it with the X to use the sample buttons.)")
elif sample_choice == "rule" and os.path.exists(RULE_SAMPLE_FILE):
    df = pd.read_csv(RULE_SAMPLE_FILE)
    st.info("Showing the rule-based sample data.")
elif sample_choice == "ml" and os.path.exists(ML_SAMPLE_FILE):
    df = pd.read_csv(ML_SAMPLE_FILE)
    st.info("Showing the ML demo data (a 2,000-row sample of the Kaggle credit card fraud dataset).")
else:
    st.info("Upload a CSV file above, or click one of the sample data buttons to get started.")
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


def highlight_fraud_prob(val):
    if val >= 0.8:
        return "background-color: #3b0d0d; color: #ff6b6b; font-weight: 600;"
    elif val >= 0.5:
        return "background-color: #3b2e0d; color: #ffd166; font-weight: 600;"
    return ""


# Check if this data matches each engine's expected format
rule_based_columns = {"amount", "merchant", "hour"}
has_rule_columns = rule_based_columns.issubset(df.columns)

expected_columns = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
has_ml_columns = all(col in df.columns for col in expected_columns)

tab1, tab2 = st.tabs(["📋 Rule-Based Detection", "🤖 ML-Based Detection"])

with tab1:
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
        st.info("This file isn't in the rule-based format (amount, merchant, hour columns), so this tab has nothing to show. Check the ML tab, or load the rule-based sample data.")


with tab2:
    st.subheader("Machine Learning Fraud Detection")

    if has_ml_columns:
        model = joblib.load("fraud_model.pkl")
        scaler = joblib.load("fraud_scaler.pkl")

        X = df.drop(columns=["Class"], errors="ignore")
        X_scaled = scaler.transform(X)

        fraud_probabilities = model.predict_proba(X_scaled)[:, 1]

        def reset_threshold():
            st.session_state.threshold_slider = RECOMMENDED_THRESHOLD

        # --- Threshold slider: how sure must the model be before flagging? ---
        threshold = st.slider(
            "Fraud probability threshold",
            min_value=0.05,
            max_value=0.99,
            value=RECOMMENDED_THRESHOLD,
            step=0.01,
            help="Only transactions scored at or above this probability are flagged. "
                 "Higher = fewer false alarms but more missed fraud.",
            key="threshold_slider"
        )
        if abs(threshold - RECOMMENDED_THRESHOLD) < 1e-9:
            st.success("✅ Recommended setting (95%): a balance between catching fraud and limiting false alarms.")
        else:
            st.warning(f"Custom setting ({threshold:.0%}). Recommended is {RECOMMENDED_THRESHOLD:.0%}.")
            st.button("Reset to recommended (95%)", on_click=reset_threshold)
        st.caption(
            "Raise the threshold to cut false alarms (analysts review fewer cases), "
            "lower it to catch more fraud. This app starts at 95%; the raw model on its own would use 50%."
        )

        df["ml_fraud_probability"] = fraud_probabilities
        df["ml_prediction"] = (fraud_probabilities >= threshold).astype(int)

        # --- Plain-English explanation of WHY the model flagged a transaction ---
        coefficients = model.coef_[0]
        feature_names = list(X.columns)

        def explain_prediction(row_position):
            contributions = coefficients * X_scaled[row_position]
            top_indices = contributions.argsort()[::-1][:3]
            top_features = [feature_names[i] for i in top_indices if contributions[i] > 0]

            if not top_features:
                return "No strongly suspicious individual features — flagged due to a subtle combined pattern."

            return f"Flagged primarily due to unusual values in: {', '.join(top_features)} (statistical pattern consistent with known fraud cases in training data)."

        # Only explain the flagged rows (much faster than explaining every row)
        flagged_positions = np.where(df["ml_prediction"].to_numpy() == 1)[0]
        ml_flagged = df.iloc[flagged_positions].copy()
        ml_flagged["ml_explanation"] = [explain_prediction(p) for p in flagged_positions]
        # Highest-probability cases first, so analysts review the likeliest fraud first
        ml_flagged = ml_flagged.sort_values("ml_fraud_probability", ascending=False)

        st.success(f"ML model ran on {len(df)} transactions.")
        col1, col2 = st.columns(2)
        col1.metric("Flagged by ML Model", len(ml_flagged))
        col2.metric("Total Transactions", len(df))

        # If the file includes the answer key (Class column), show how the model did
        if "Class" in df.columns:
            actual_fraud = int((df["Class"] == 1).sum())
            caught = int(((df["Class"] == 1) & (df["ml_prediction"] == 1)).sum())
            false_alarms = int(((df["Class"] == 0) & (df["ml_prediction"] == 1)).sum())

            st.markdown("**How the model did on this file** (this file includes the true fraud labels)")
            res1, res2, res3 = st.columns(3)
            res1.metric("Actual fraud in file", actual_fraud)
            res2.metric("Fraud caught", f"{caught} of {actual_fraud}")
            res3.metric("False alarms", false_alarms)

        if len(ml_flagged) == 0:
            st.info("No transactions are flagged at this threshold. Lower the slider to flag more.")
        else:
            st.write("Transactions flagged by the ML model (highest fraud probability first):")

            for idx, row in ml_flagged.head(20).iterrows():
                with st.container(border=True):
                    st.write(f"**Transaction (row {idx})** — Fraud probability: {row['ml_fraud_probability']:.1%}")
                    st.write(row["ml_explanation"])

            if len(ml_flagged) > 20:
                st.caption(f"Showing the top 20 of {len(ml_flagged)} flagged transactions as cards. The table below lists up to 1,000.")

            display_columns = ["Amount", "ml_fraud_probability", "ml_explanation"]
            styled_ml = ml_flagged[display_columns].head(1000).style.map(
                highlight_fraud_prob, subset=["ml_fraud_probability"]
            )
            st.dataframe(styled_ml, use_container_width=True)

    else:
        st.info("This file doesn't have the columns the ML model needs (Time, V1–V28, Amount). Download the ML demo sample from the sidebar, or click 'Try ML demo data' above.")
        