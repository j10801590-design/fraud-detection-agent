import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the real dataset
df = pd.read_csv("creditcard.csv")

# Features (inputs) and target (what we're predicting)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# --- NEW: Scale the features ---
# StandardScaler transforms each column so it has mean=0 and standard deviation=1.
# This puts Amount and the V1-V28 columns on a comparable numeric scale,
# which helps Logistic Regression converge properly.
scaler = StandardScaler()

# Fit the scaler on training data only, then transform both train and test.
# (We "fit" only on training data to avoid letting test data leak into training.)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model on the scaled data
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

print("\nModel trained!")

# Test it on the hidden test set
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save both the model AND the scaler — we'll need the scaler later too,
# since any new data must be scaled the same way before predicting
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")
print("\nModel and scaler saved.")