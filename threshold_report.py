import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Full Kaggle file (stays on your computer)
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# SAME split settings as train_model.py -> the held-out rows the model never trained on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")
probs = model.predict_proba(scaler.transform(X_test))[:, 1]

total_fraud = int((y_test == 1).sum())
print(f"Held-out test rows: {len(y_test)} | actual fraud: {total_fraud}")
print()
print("Cutoff | Fraud caught | False alarms | Precision")
for t in [0.50, 0.75, 0.90, 0.95, 0.99]:
    flagged = probs >= t
    caught = int(((y_test.values == 1) & flagged).sum())
    false_alarms = int(((y_test.values == 0) & flagged).sum())
    precision = caught / max(caught + false_alarms, 1)
    print(f"{t:.0%}    | {caught} of {total_fraud}      | {false_alarms}          | {precision:.1%}")
    