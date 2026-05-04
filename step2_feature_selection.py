# ============================================================
# STEP 2 - FEATURE SELECTION & TRANSFORMATION
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# - Removes data leakage columns
# - Removes low-importance features (PDW, LYMn, NEUTn, PCT)
# - Displays feature correlation with target
# - Saves selected features list
# ============================================================

import pandas as pd
import numpy as np

# ------------------ Load ------------------

print("=" * 55)
print(" Ensemble Learning - Anemia Severity Prediction")
print("=" * 55)
print("\n[STEP 2] Loading preprocessed dataset...")

df = pd.read_csv("data/anemia_with_severity.csv")

print(f"  Shape: {df.shape}")

# ------------------ Drop Leakage & Unwanted Columns ------------------

print("\n[STEP 2] Removing leakage and unwanted columns...")

# PDW   → clinically less relevant for severity prediction
# LYMn  → removed (low importance / not needed)
# NEUTn → removed (low importance / not needed)
# PCT   → removed (low importance / not needed)
DROP_COLS = ["Diagnosis", "Severity", "Severity_Label", "PDW", "LYMn", "NEUTn", "PCT"]

X = df.drop(columns=DROP_COLS)
y = df["Severity_Label"]

print(f"  Dropped columns : {DROP_COLS}")
print(f"  Remaining features ({len(X.columns)}): {X.columns.tolist()}")

# ------------------ Feature Correlation ------------------

print("\n[STEP 2] Feature correlation with Severity Label:")

corr = df[X.columns.tolist() + ["Severity_Label"]].corr()["Severity_Label"].drop("Severity_Label")
corr_sorted = corr.abs().sort_values(ascending=False)

for feat, val in corr_sorted.items():
    print(f"  {feat:<12} : {val:.4f}")

# ------------------ Save Feature List ------------------

selected_features = X.columns.tolist()

with open("data/selected_features.txt", "w") as f:
    for feat in selected_features:
        f.write(feat + "\n")

print("\n  [SUCCESS] Selected features saved → data/selected_features.txt")
print("=" * 55)