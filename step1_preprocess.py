# ============================================================
# STEP 1 - DATA PREPROCESSING
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# - Loads dataset from reliable clinical CBC data source
# - Handles missing values
# - Normalizes features
# - Adds Severity labels based on HGB values
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# ------------------ Load Dataset ------------------

print("=" * 55)
print(" Ensemble Learning - Anemia Severity Prediction")
print("=" * 55)
print("\n[STEP 1] Loading dataset...")

df = pd.read_csv("data/diagnosed_cbc_data_v4.csv")

print(f"  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n  First 5 rows:")
print(df.head())

# ------------------ Handle Missing Values ------------------

print("\n[STEP 1] Handling missing values...")

missing = df.isnull().sum()
print(f"  Missing values per column:\n{missing[missing > 0]}")

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("  Missing values filled with column median.")

# ------------------ Anemia Severity Labeling ------------------

print("\n[STEP 1] Assigning Anemia Severity based on HGB levels...")

def anemia_severity(hgb):
    """
    Clinical severity classification based on WHO guidelines:
    - Normal  : HGB >= 12 g/dL
    - Mild    : HGB 10–11.9 g/dL
    - Moderate: HGB 7–9.9 g/dL
    - Severe  : HGB < 7 g/dL
    """
    if hgb >= 12:
        return "Normal"
    elif hgb >= 10:
        return "Mild"
    elif hgb >= 7:
        return "Moderate"
    else:
        return "Severe"

df["Severity"] = df["HGB"].apply(anemia_severity)

severity_map = {
    "Normal"  : 0,
    "Mild"    : 1,
    "Moderate": 2,
    "Severe"  : 3
}

df["Severity_Label"] = df["Severity"].map(severity_map)

print("  Severity distribution:")
print(df["Severity"].value_counts())

# ------------------ Normalization ------------------

print("\n[STEP 1] Normalizing numeric features (MinMaxScaler)...")

exclude_cols = ["Diagnosis", "Severity", "Severity_Label"]
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("  Normalization complete.")

# Save scaler for use in prediction
import os
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/scaler_columns.pkl", "wb") as f:
    pickle.dump(list(feature_cols), f)

print("  Scaler saved → models/scaler.pkl")

# ------------------ Save ------------------

df.to_csv("data/anemia_with_severity.csv", index=False)

print("\n  [SUCCESS] Saved → data/anemia_with_severity.csv")
print("=" * 55)
