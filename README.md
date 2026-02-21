# Ensemble Learning Model for Multi-Level Anemia Severity Prediction

## Project Structure

```
anemia_project/
│
├── data/
│   └── diagnosed_cbc_data_v4.csv       ← Place your dataset here
│
├── models/                             ← Saved model stored here automatically
│
├── step1_preprocess.py                 ← Data cleaning, normalization, labeling
├── step2_feature_selection.py          ← Feature correlation & selection
├── step3_train_model.py                ← SMOTE + Ensemble model training
├── step4_evaluate.py                   ← Accuracy, Precision, Recall, F1, CM
├── step5_predict.py                    ← User input → Prediction + Confidence
├── main.py                             ← Runs all steps in order
└── README.md
```

## How to Run

### Option 1 — Run all steps at once
```bash
python main.py
```

### Option 2 — Run each step separately
```bash
python step1_preprocess.py
python step2_feature_selection.py
python step3_train_model.py
python step4_evaluate.py
python step5_predict.py
```

## Requirements

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost
```

## Pipeline Overview

| Step | File | Description |
|------|------|-------------|
| 1 | step1_preprocess.py | Load dataset, handle missing values, normalize, assign severity labels |
| 2 | step2_feature_selection.py | Remove leakage columns, select important clinical features |
| 3 | step3_train_model.py | Apply SMOTE, train RF + AdaBoost + XGBoost ensemble |
| 4 | step4_evaluate.py | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| 5 | step5_predict.py | User CBC input → Severity prediction + Confidence score + Diet |

## Models Used

- **Random Forest** — 200 trees, robust to overfitting
- **AdaBoost** — Boosting weak learners
- **XGBoost** — Gradient boosting for high accuracy
- **VotingClassifier (soft)** — Combines all three using probability averaging

## Severity Classification (WHO Guidelines)

| Label | HGB Level | Code |
|-------|-----------|------|
| Normal | ≥ 12 g/dL | 0 |
| Mild | 10–11.9 g/dL | 1 |
| Moderate | 7–9.9 g/dL | 2 |
| Severe | < 7 g/dL | 3 |
