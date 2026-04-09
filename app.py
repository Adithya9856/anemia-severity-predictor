# ============================================================
# STREAMLIT UI - WHITE THEME
# Project: Ensemble Learning Model for Multi-Level Anemia Severity Prediction
# ============================================================

import streamlit as st
import pandas as pd
import pickle

# ------------------ Page Config ------------------

st.set_page_config(
    page_title="Anemia Severity Predictor",
    page_icon="🩸",
    layout="wide"
)

# ------------------ Custom CSS ------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1a1a1a;
}

[data-testid="stAppViewContainer"] {
    background: #ffffff;
}

[data-testid="stHeader"] {
    background-color: #ffffff;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
}

.hero h1 {
    font-size: 3.2rem;
    background: linear-gradient(135deg, #e85d5d, #c0392b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero p {
    color: #555555;
    font-size: 1.1rem;
    font-weight: 300;
}

.card {
    background: #f9f9f9;
    border: 1px solid rgba(232,93,93,0.3);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
}

.result-card {
    background: linear-gradient(135deg, rgba(232,93,93,0.08), rgba(255,154,154,0.03));
    border: 1px solid rgba(232,93,93,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-card h2 {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}

.severity-normal   { color: #2e7d32; }
.severity-mild     { color: #e65100; }
.severity-moderate { color: #c62828; }
.severity-severe   { color: #6a1b9a; }

.diet-item {
    background: #fff5f5;
    border-left: 3px solid #e85d5d;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    color: #1a1a1a;
}

.badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

.badge-normal   { background: rgba(46,125,50,0.1);   color: #2e7d32; border: 1px solid #2e7d32; }
.badge-mild     { background: rgba(230,81,0,0.1);    color: #e65100; border: 1px solid #e65100; }
.badge-moderate { background: rgba(198,40,40,0.1);   color: #c62828; border: 1px solid #c62828; }
.badge-severe   { background: rgba(106,27,154,0.1);  color: #6a1b9a; border: 1px solid #6a1b9a; }

.stButton > button {
    background: linear-gradient(135deg, #e85d5d, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(232,93,93,0.3) !important;
}

.warning-box {
    background: rgba(106,27,154,0.08);
    border: 1px solid #6a1b9a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    color: #6a1b9a;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #e85d5d;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(232,93,93,0.2);
    padding-bottom: 0.5rem;
}

.normal-range {
    font-size: 0.75rem;
    color: #999999;
    margin-top: 0.2rem;
}

.info-card {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #555555;
    line-height: 1.8;
}

div[data-testid="column"] { padding: 0 0.5rem; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
def load_accuracy():
    try:
        with open("models/accuracy.txt", "r") as f:
            return f.read()
    except:
        return "Not Available"
@st.cache_resource
def load_model():
    with open("models/ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/selected_features.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]
    return model, scaler, features

try:
    ensemble, scaler, selected_features = load_model()
    accuracy = load_accuracy()   
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠ Model not found. Please run step1 → step3 first.\n\n{e}")

# ------------------ Constants ------------------

LABELS  = {0: "Normal", 1: "Mild Anemia", 2: "Moderate Anemia", 3: "Severe Anemia"}
CLASSES = {0: "normal", 1: "mild",        2: "moderate",        3: "severe"}

DIET = {
    0: ["Balanced meals with all food groups", "Fresh fruits and vegetables", "Whole grains and legumes", "Stay well hydrated"],
    1: ["Spinach, kale, and leafy greens", "Dates, raisins, and dried fruits", "Beetroot and pomegranate juice", "Eggs and Vitamin C rich foods", "Fortified breakfast cereals"],
    2: ["Lentils, beans, and chickpeas", "Pomegranate and beetroot daily", "Jaggery-based foods", "Fish (especially tuna, sardines)", "Consult a doctor for iron supplements"],
    3: ["Immediate doctor consultation required", "Chicken liver and red meat", "Iron-fortified cereals and bread", "Beans, lentils, and tofu", "Vitamin C to boost iron absorption", "Medical treatment may be needed"]
}

NORMAL_RANGES = {
    "WBC"  : "4.0 – 11.0 ×10³/µL",
    "LYMp" : "20 – 40 %",
    "NEUTp": "50 – 70 %",
    "LYMn" : "1.0 – 4.8 ×10³/µL",
    "NEUTn": "1.8 – 7.7 ×10³/µL",
    "RBC"  : "4.2 – 5.4 ×10⁶/µL",
    "HCT"  : "36 – 48 %",
    "MCV"  : "80 – 100 fL",
    "MCH"  : "27 – 33 pg",
    "MCHC" : "32 – 36 g/dL",
    "PLT"  : "150 – 400 ×10³/µL",
    "PCT"  : "0.15 – 0.40 %"
}

DEFAULT_VALUES = {
    "WBC": 0.0, "LYMp": 0.0, "NEUTp": 0.0, "LYMn": 0.0,
    "NEUTn": 0.0, "RBC": 0.0, "HCT": 0.0, "MCV": 0.0,
    "MCH": 0.0, "MCHC": 0.0, "PLT": 0.0, "PCT": 0.0
}

# ------------------ Hero ------------------

st.markdown("""
<div class="hero">
    <h1>🩸 Anemia Severity Predictor</h1>
    <p>Ensemble Learning Model · Random Forest + AdaBoost + XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Layout ------------------

if model_loaded:
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Enter CBC Values</div>', unsafe_allow_html=True)

        inputs = {}
        feature_list = selected_features
        rows = [feature_list[i:i+2] for i in range(0, len(feature_list), 2)]

        for row in rows:
            cols = st.columns(2)
            for i, feat in enumerate(row):
                with cols[i]:
                    inputs[feat] = st.number_input(
                        feat,
                        value=DEFAULT_VALUES.get(feat, 0.0),
                        step=0.01,
                        format="%.2f",
                        key=feat
                    )
                    st.markdown(f'<div class="normal-range">Normal: {NORMAL_RANGES.get(feat, "—")}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔬 Predict Anemia Severity")

    with right_col:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            features_vals = [inputs[f] for f in selected_features]
            all_scaler_cols = list(scaler.feature_names_in_)
            input_full = pd.DataFrame([[0]*len(all_scaler_cols)], columns=all_scaler_cols)

            for col, val in zip(selected_features, features_vals):
                if col in input_full.columns:
                    input_full[col] = val

            input_scaled = pd.DataFrame(scaler.transform(input_full), columns=all_scaler_cols)
            input_df = input_scaled[selected_features]

            pred  = ensemble.predict(input_df)[0]
            label = LABELS[pred]
            cls   = CLASSES[pred]

            st.markdown(f"""
            <div class="result-card">
                <div class="badge badge-{cls}">{cls.upper()}</div>
                <h2 class="severity-{cls}">{label}</h2>
                <p style="color:#777777; font-size:0.95rem;">Based on your CBC values</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:1.5rem;">Diet Recommendations</div>', unsafe_allow_html=True)
            for item in DIET[pred]:
                st.markdown(f'<div class="diet-item">✔ {item}</div>', unsafe_allow_html=True)

            if pred == 3:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>Severe Anemia Detected</strong><br>
                    Please consult a medical professional immediately.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 3rem 2rem; color:#aaaaaa;">
                <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
                <p style="font-size:1rem; color:#888888;">Enter CBC values on the left<br>and click <strong style="color:#e85d5d;">Predict</strong> to see results.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <strong style="color:#e85d5d;">About this model</strong><br>
            Ensemble of Random Forest, AdaBoost & XGBoost<br>
            Trained on 1481 clinical CBC records<br>
            SMOTE applied for class balancing<br>
            Test Accuracy: <strong style="color:#1a1a1a;">{accuracy}</strong>
        </div>
        """, unsafe_allow_html=True)