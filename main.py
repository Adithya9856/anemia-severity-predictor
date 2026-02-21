# ============================================================
# MAIN - RUN FULL PIPELINE
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================
# Run this file to execute all steps in order:
#   Step 1 → Preprocess data
#   Step 2 → Feature selection
#   Step 3 → Train model
#   Step 4 → Evaluate model
#   Step 5 → Predict (user input)
# ============================================================

import subprocess
import sys

steps = [
    ("STEP 1 - Data Preprocessing"       , "step1_preprocess.py"),
    ("STEP 2 - Feature Selection"         , "step2_feature_selection.py"),
    ("STEP 3 - Model Training"            , "step3_train_model.py"),
    ("STEP 4 - Model Evaluation"          , "step4_evaluate.py"),
    ("STEP 5 - Prediction (User Input)"   , "step5_predict.py"),
]

print("\n" + "=" * 55)
print("  Anemia Severity Prediction - Full Pipeline")
print("=" * 55 + "\n")

for title, script in steps:
    print(f"\n{'='*55}")
    print(f"  Running: {title}")
    print(f"{'='*55}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n!! Error in {script}. Pipeline stopped.")
        sys.exit(1)

print("\n" + "=" * 55)
print("  ✔  Full pipeline completed successfully!")
print("=" * 55)
