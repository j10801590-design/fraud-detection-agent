import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full Kaggle dataset (stays on your computer, never pushed to GitHub)
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# SAME split settings as train_model.py, so this gives the exact same held-out test rows
# (rows the model never saw during training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df["Class"] = y_test

# Keep every fraud case from the test set, plus a random sample of normal transactions
fraud = test_df[test_df["Class"] == 1]
normal = test_df[test_df["Class"] == 0].sample(n=2000 - len(fraud), random_state=42)

# Mix them together and save
demo = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
demo.to_csv("demo_creditcard_sample.csv", index=False)

print("Saved demo_creditcard_sample.csv")
print("Rows:", len(demo))
print("Fraud rows:", int(demo["Class"].sum()))
