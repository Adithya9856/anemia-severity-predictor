# ============================================================
# STEP 4 - MODEL EVALUATION
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# - Loads saved model and test data
# - Evaluates: Accuracy, Precision, Recall, F1-Score
# - Displays Confusion Matrix
# ============================================================

import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------ Load Model & Test Data ------------------

print("=" * 55)
print(" Ensemble Learning - Anemia Severity Prediction")
print("=" * 55)
print("\n[STEP 4] Loading model and test data...")

with open("models/ensemble_model.pkl", "rb") as f:
    ensemble = pickle.load(f)

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

print(f"  Test samples loaded: {X_test.shape[0]}")

# ------------------ Predict ------------------

print("\n[STEP 4] Running predictions on test data...")

y_pred = ensemble.predict(X_test)

# ------------------ Accuracy ------------------

acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print(f"  TEST ACCURACY : {acc * 100:.2f}%")
print("=" * 55)

# ------------------ Classification Report ------------------

label_names = ["Normal", "Mild", "Moderate", "Severe"]

print("\n[STEP 4] Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names))

# ------------------ Confusion Matrix ------------------

print("[STEP 4] Confusion Matrix:\n")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index  =[f"Actual {l}"    for l in label_names],
    columns=[f"Predicted {l}" for l in label_names]
)
print(cm_df)

print("\n  [SUCCESS] Evaluation complete.")
print("=" * 55)
