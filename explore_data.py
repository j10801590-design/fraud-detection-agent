import pandas as pd

# Load the real dataset
df = pd.read_csv("creditcard.csv")

# Basic shape: how many rows (transactions) and columns
print("Shape (rows, columns):", df.shape)

# Show the column names
print("\nColumns:", list(df.columns))

# Show the first few rows so we can see what the data looks like
print("\nFirst 5 rows:")
print(df.head())

# Count how many are fraud (Class=1) vs normal (Class=0)
print("\nFraud vs Normal counts:")
print(df["Class"].value_counts())

# What percentage of transactions are fraud?
fraud_percentage = (df["Class"].sum() / len(df)) * 100
print(f"\nPercentage that are fraud: {fraud_percentage:.4f}%")

# Basic stats on the Amount column
print("\nAmount statistics:")
print(df["Amount"].describe())
