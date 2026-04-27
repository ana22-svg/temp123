"""
app.py
------
Flask REST API for the Loan Risk Prediction System.
Loads pre-trained models saved by train_model.py.

Run:   python app.py
Test:  POST http://127.0.0.1:5000/predict
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load all saved artifacts
# ─────────────────────────────────────────────
print("[app] Loading models...")

regressor  = joblib.load("model/regressor.pkl")
classifier = joblib.load("model/classifier.pkl")
scaler     = joblib.load("model/scaler.pkl")
le_grade   = joblib.load("model/le_grade.pkl")
le_target  = joblib.load("model/le_target.pkl")

with open("model/training_stats.json") as f:
    training_stats = json.load(f)

LOAN_AMNT_MEAN     = training_stats["loan_amnt_mean"]
INSTALLMENT_MEDIAN = training_stats["installment_median"]
DTI_THRESHOLD      = training_stats["dti_high_threshold"]

print("[app] ✓ All models loaded. Server ready.\n")


# ─────────────────────────────────────────────
# Reusable preprocessing (mirrors preprocess.py logic)
# ─────────────────────────────────────────────

def categorize_risk(score: float) -> str:
    """Same thresholds as Assignment 5."""
    if score < 33:
        return "Low"
    elif score < 66:
        return "Medium"
    else:
        return "High"


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Converts raw JSON input into a scaled feature array.
    Applies the same pipeline as train_model.py:
      encoding → feature engineering → scaling
    """
    loan_amnt   = float(data["loan_amnt"])
    installment = float(data["installment"])
    dti         = float(data["dti"])
    
    if "grade_encoded" in data:
        grade_encoded = int(data["grade_encoded"])
    else:
        grade_str   = str(data["grade"]).upper()
        if grade_str not in le_grade.classes_:
            raise ValueError(f"Unknown grade '{grade_str}'. Valid grades: {list(le_grade.classes_)}")
        grade_encoded = int(le_grade.transform([grade_str])[0])

    # Feature engineering (same logic as training)
    is_high_loan        = 1 if loan_amnt   > LOAN_AMNT_MEAN     else 0
    is_high_installment = 1 if installment > INSTALLMENT_MEDIAN  else 0
    is_high_dti         = 1 if dti         > DTI_THRESHOLD        else 0
    loan_burden         = loan_amnt / installment if installment != 0 else 0
    risk_flag           = 1 if (is_high_loan == 1 and is_high_dti == 1) else 0

    features = np.array([[
        loan_amnt, installment, dti, grade_encoded,
        is_high_loan, is_high_installment, is_high_dti,
        loan_burden, risk_flag
    ]])

    return scaler.transform(features)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Loan Risk Prediction API",
        "endpoints": {
            "/predict":       "POST — predict risk for a single borrower",
            "/predict/batch": "POST — predict risk for multiple borrowers",
            "/health":        "GET  — server health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


# ── Single Prediction ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Input JSON:
    {
        "loan_amnt":   15000,
        "installment": 450.5,
        "dti":         18.5,
        "grade":       "C"
    }

    Response:
    {
        "risk_score":           61.42,
        "risk_category_reg":   "Medium",    ← from regressor
        "risk_category_clf":   "Medium",    ← from classifier (more reliable)
        "class_probabilities": {
            "High":   0.12,
            "Low":    0.21,
            "Medium": 0.67
        }
    }
    """
    # ── Validate input ──
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["loan_amnt", "installment", "dti"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    if "grade" not in data and "grade_encoded" not in data:
        return jsonify({"error": "Missing field: grade or grade_encoded"}), 400

    try:
        X_scaled = build_feature_vector(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500

    # ── Regressor prediction ──
    reg_score    = float(np.clip(regressor.predict(X_scaled)[0], 0, 100))
    reg_category = categorize_risk(reg_score)

    # ── Classifier prediction ──
    clf_label    = int(classifier.predict(X_scaled)[0])
    clf_category = str(le_target.inverse_transform([clf_label])[0])
    clf_probs    = classifier.predict_proba(X_scaled)[0]
    prob_dict    = {
        cls: round(float(p), 4)
        for cls, p in zip(le_target.classes_, clf_probs)
    }

    return jsonify({
        "input":                data,
        "risk_score":           round(reg_score, 2),
        "risk_category":        clf_category,
        "risk_category_reg":    reg_category,
        "risk_category_clf":    clf_category,
        "class_probabilities":  prob_dict
    })


# ── Batch Prediction ───────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Input JSON:
    {
        "borrowers": [
            {"loan_amnt": 15000, "installment": 450, "dti": 18.5, "grade": "C"},
            {"loan_amnt": 8000,  "installment": 200, "dti": 10.0, "grade": "A"}
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data or "borrowers" not in data:
        return jsonify({"error": "Provide a 'borrowers' list"}), 400

    predictions = []
    for i, borrower in enumerate(data["borrowers"]):
        try:
            X_scaled     = build_feature_vector(borrower)
            reg_score    = float(np.clip(regressor.predict(X_scaled)[0], 0, 100))
            clf_label    = int(classifier.predict(X_scaled)[0])
            clf_category = str(le_target.inverse_transform([clf_label])[0])
            clf_probs    = classifier.predict_proba(X_scaled)[0]

            predictions.append({
                "index":               i,
                "input":               borrower,
                "risk_score":          round(reg_score, 2),
                "risk_category":       clf_category,
                "class_probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(le_target.classes_, clf_probs)
                }
            })
        except Exception as e:
            predictions.append({"index": i, "error": str(e)})

    return jsonify({
        "total":       len(data["borrowers"]),
        "predictions": predictions
    })


# ─────────────────────────────────────────────
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Loan Risk API on port {port}")
    app.run(host="0.0.0.0", port=port)
