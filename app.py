import streamlit as st
import pandas as pd

st.title("AI Transaction Risk & Fraud Investigation Agent")
st.write("This dashboard uses simulated transaction data for demonstration purposes only.")

# Load the CSV file
df = pd.read_csv("sample_transactions.csv")

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

df[["flag_reasons", "risk_score"]] = df.apply(evaluate_transaction, axis=1)

def score_to_level(score):
    if score >= 50:
        return "High"
    elif score >= 20:
        return "Medium"
    else:
        return "Low"

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

# --- NEW: Analyst review section ---
st.subheader("Investigate Flagged Transactions")

# st.session_state is a special dictionary Streamlit gives us that
# PERSISTS across reruns (normally, every click/interaction wipes
# out regular variables and reruns the whole script from top to bottom).
# We use it here to remember each transaction's review status even
# as the page re-runs when you interact with a dropdown.

# "if X not in st.session_state" means "only set this up ONCE,
# the first time the app runs — don't reset it every rerun."
if "review_status" not in st.session_state:
    # Create a dictionary like {1: "Not Reviewed", 2: "Not Reviewed", ...}
    # one entry per transaction_id, all starting as "Not Reviewed"
    st.session_state.review_status = {
        txn_id: "Not Reviewed" for txn_id in df["transaction_id"]
    }

# Only show flagged (Medium/High) transactions for review
flagged_df = df[df["risk_level"] != "Low"]

# Loop through each flagged transaction one at a time
for _, row in flagged_df.iterrows():
    with st.container(border=True):
        st.write(f"**Transaction #{row['transaction_id']}** — {row['merchant']}")
        st.write(f"Amount: ${row['amount']:.2f} | Hour: {row['hour']} | Risk: {row['risk_level']} (score: {row['risk_score']})")
        st.write(f"Reasons: {', '.join(row['flag_reasons'])}")

        # A dropdown for the analyst to set the review status.
        # key=... gives this specific dropdown a unique name so Streamlit
        # doesn't confuse it with the other transactions' dropdowns.
        selected_status = st.selectbox(
            "Review status",
            options=["Not Reviewed", "Confirmed Fraud", "False Positive"],
            index=["Not Reviewed", "Confirmed Fraud", "False Positive"].index(
                st.session_state.review_status[row["transaction_id"]]
            ),
            key=f"status_{row['transaction_id']}"
        )

        # Save whatever the analyst picked back into session_state
        st.session_state.review_status[row["transaction_id"]] = selected_status
        # --- NEW: Export report ---
st.subheader("Export Report")

# Build a copy of flagged_df that includes the review status
export_df = flagged_df.copy()
export_df["review_status"] = export_df["transaction_id"].map(st.session_state.review_status)

# Convert the DataFrame to CSV text (in memory, not saved to disk)
csv_data = export_df.to_csv(index=False)

st.download_button(
    label="Download Flagged Transactions Report (CSV)",
    data=csv_data,
    file_name="flagged_transactions_report.csv",
    mime="text/csv"
)
