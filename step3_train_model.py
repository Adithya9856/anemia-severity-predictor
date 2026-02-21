# ============================================================
# STEP 3 - MODEL TRAINING
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# - Applies SMOTE only on training data (correct approach)
# - Trains ensemble: Random Forest + AdaBoost + XGBoost
# - Saves trained model to models/ensemble_model.pkl
# ============================================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------ Load ------------------

print("=" * 55)
print(" Ensemble Learning - Anemia Severity Prediction")
print("=" * 55)
print("\n[STEP 3] Loading dataset...")

df = pd.read_csv("data/anemia_with_severity.csv")

# Load selected features
with open("data/selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

X = df[selected_features]
y = df["Severity_Label"]

print(f"  Features used : {selected_features}")
print(f"  Dataset shape : {X.shape}")

# ------------------ Train / Test Split ------------------

print("\n[STEP 3] Splitting dataset (80% train / 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Train samples : {X_train.shape[0]}")
print(f"  Test  samples : {X_test.shape[0]}")

# Save test split for evaluation step
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# ------------------ SMOTE (only on training data) ------------------

print("\n[STEP 3] Applying SMOTE to balance training classes...")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"  After SMOTE - Train samples: {X_train_res.shape[0]}")
print(f"  Class distribution:\n  {pd.Series(y_train_res).value_counts().to_dict()}")

# ------------------ Define Models ------------------

print("\n[STEP 3] Defining ensemble models...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

ada = AdaBoostClassifier(
    random_state=42
)

xgb = XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False
)

ensemble = VotingClassifier(
    estimators=[
        ('RandomForest', rf),
        ('AdaBoost'    , ada),
        ('XGBoost'     , xgb)
    ],
    voting='soft'
)

# ------------------ Train ------------------

print("\n[STEP 3] Training ensemble model...")
print("  (This may take a moment...)\n")

ensemble.fit(X_train_res, y_train_res)

print("  Training complete!")

# ------------------ Save Model ------------------

import os
os.makedirs("models", exist_ok=True)

with open("models/ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

print("\n  [SUCCESS] Model saved → models/ensemble_model.pkl")
print("=" * 55)
