# ============================================================
# STEP 5 - PREDICTION WITH CONFIDENCE SCORE
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# - Loads trained ensemble model
# - Takes user CBC input
# - Normalizes input using saved scaler
# - Predicts Anemia Severity
# - Shows Confidence Score (%)
# - Provides Diet Recommendations
# ============================================================

import pandas as pd
import pickle

# ------------------ Load Model & Features ------------------

print("=" * 55)
print(" Ensemble Learning - Anemia Severity Prediction")
print("=" * 55)

with open("models/ensemble_model.pkl", "rb") as f:
    ensemble = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("data/selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

# ------------------ Labels ------------------

LABELS = {
    0: "Normal",
    1: "Mild Anemia",
    2: "Moderate Anemia",
    3: "Severe Anemia"
}

# ------------------ Diet Recommendations ------------------

DIET = {
    0: [
        "Balanced meals with all food groups",
        "Fresh fruits and vegetables",
        "Whole grains and legumes",
        "Stay well hydrated"
    ],
    1: [
        "Spinach, kale, and leafy greens",
        "Dates, raisins, and dried fruits",
        "Beetroot and pomegranate juice",
        "Eggs and Vitamin C rich foods",
        "Fortified breakfast cereals"
    ],
    2: [
        "Lentils, beans, and chickpeas",
        "Pomegranate and beetroot daily",
        "Jaggery-based foods",
        "Fish (especially tuna, sardines)",
        "Consult a doctor for iron supplements"
    ],
    3: [
        "Immediate doctor consultation required",
        "Chicken liver and red meat",
        "Iron-fortified cereals and bread",
        "Beans, lentils, and tofu",
        "Vitamin C to boost iron absorption",
        "Medical treatment may be needed"
    ]
}

# ------------------ User Input ------------------

print("\n[STEP 5] Enter CBC (Complete Blood Count) Values:\n")

features = []

for col in selected_features:
    while True:
        try:
            val = float(input(f"  {col}: "))
            features.append(val)
            break
        except ValueError:
            print("  !! Please enter a valid numeric value.")

# ------------------ Normalize Input ------------------

# Create full dataframe with all columns the scaler expects, filled with 0
all_scaler_cols = list(scaler.feature_names_in_)
input_full = pd.DataFrame([[0] * len(all_scaler_cols)], columns=all_scaler_cols)

# Fill in the values we have
for col, val in zip(selected_features, features):
    if col in input_full.columns:
        input_full[col] = val

# Scale
input_scaled_full = scaler.transform(input_full)
input_scaled_df = pd.DataFrame(input_scaled_full, columns=all_scaler_cols)

# Keep only selected features for model
input_df = input_scaled_df[selected_features]

# ------------------ Prediction ------------------

print("\n[STEP 5] Analyzing...")

pred       = ensemble.predict(input_df)[0]
proba      = ensemble.predict_proba(input_df)[0]
confidence = proba[pred] * 100

# ------------------ Output ------------------

print("\n" + "=" * 55)
print("  ANEMIA SEVERITY PREDICTION RESULT")
print("=" * 55)
print(f"\n  Predicted Severity  : {LABELS[pred]}")
print(f"  Confidence Score    : {confidence:.2f}%")

print("\n" + "=" * 55)
print("  DIET RECOMMENDATIONS")
print("=" * 55)

for item in DIET[pred]:
    print(f"  ✔  {item}")

print("\n" + "=" * 55)

if pred == 3:
    print("  ⚠  WARNING: Severe anemia detected.")
    print("     Please consult a medical professional immediately.")
    print("=" * 55)
