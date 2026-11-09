# AI Transaction Risk & Fraud Investigation Agent

A Streamlit web app that helps a (simulated) bank fraud analyst identify, score, and investigate suspicious transactions — from raw CSV data to a reviewable, exportable report.

> **Note:** This project uses simulated/sample transaction data only. No real financial or customer data is used or represented. Flagged transactions are for demonstration purposes only and are never presented as real fraud cases.
**🔗 Live demo:** https://fraud-investigation-agent.streamlit.app/


## Features

- Loads transaction data from CSV
- Rule-based fraud detection engine (large amounts, unusual hours, vague merchant names)
- Numeric risk scoring mapped to Low / Medium / High risk levels
- Dashboard with summary metrics and a risk-level chart
- Per-transaction analyst review workflow (Not Reviewed / Confirmed Fraud / False Positive), persisted via Streamlit session state
- Exportable CSV report of flagged transactions and review decisions

## Tech Stack

- Python
- Streamlit
- pandas

## How to Run Locally

1. Clone this repo:
```
git clone https://github.com/j10801590-design/fraud-detection-agent.git
cd fraud-detection-agent
```

2. Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
streamlit run app.py
```

## Roadmap

- [x] Let analysts upload their own CSV files
- [ ] Swap in a larger, real public fraud dataset (e.g., Kaggle Credit Card Fraud Detection) for realistic-scale evaluation
- [x] Add a scikit-learn machine learning model alongside the rule-based engine
- [ ] Simulate live transaction events (webhook-style) so new transactions can appear in real time

## About This Project

Built as a portfolio project to learn end-to-end app development: data processing, rule-based logic, interactive dashboards, and eventually machine learning — while keeping the code understandable and explainable.

## Results

The ML model is a logistic regression trained on the public Kaggle Credit Card Fraud dataset (284,807 transactions, 492 fraud cases). I tested it on 56,962 transactions it never saw during training (98 fraud cases).

The app has an adjustable fraud-probability threshold, because lowering false alarms and catching more fraud pull in opposite directions:

| Threshold | Fraud caught | False alarms |
|-----------|--------------|--------------|
| 50% (default) | 90 of 98 | 1,386 |
| 75% | 88 of 98 | 539 |
| 95% (app default) | 87 of 98 | 136 |

I set 95% as the default because it cut false alarms by about 90% while missing only 3 more fraud cases. Analysts can change it with the slider depending on how many alerts they can review. With only 98 fraud cases in the test set, small differences are noisy.

## Demo data

The ML demo sample (`demo_creditcard_sample.csv`) has 2,000 transactions taken from the held-out test split, with fraud over-represented so the demo is easy to see. The full dataset is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group / Worldline).
