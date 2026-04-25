"""
preprocess.py
-------------
Handles all data loading, cleaning, encoding, and feature engineering.
Called by both train_model.py (for training) and app.py (for live input).
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ─────────────────────────────────────────────
# STEP 1: CSV → JSON → DataFrame  (Assignment 5 ETL)
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Replicates the Assignment 5 ETL pipeline:
    CSV  →  JSON (saved to disk)  →  DataFrame
    """
    # Load CSV
    df_csv = pd.read_csv(csv_path)

    # Convert to JSON and save (demonstrates ETL pipeline)
    json_path = csv_path.replace(".csv", ".json")
    df_csv.to_json(json_path, orient="records", lines=True)
    print(f"[ETL] Saved JSON to: {json_path}")

    # Reload from JSON back to DataFrame
    df = pd.read_json(json_path, lines=True)
    print(f"[ETL] Loaded DataFrame from JSON — shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 2: Missing Value Handling  (Assignment 4 & 5)
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numerical  → mean (loan_amnt) / median (installment, dti)
    Categorical → mode (grade)
    """
    df = df.copy()

    before = df.isnull().sum().sum()

    # Numerical
    df["loan_amnt"]   = df["loan_amnt"].fillna(df["loan_amnt"].mean())
    df["installment"] = df["installment"].fillna(df["installment"].median())
    df["dti"]         = df["dti"].fillna(df["dti"].median())

    # Categorical
    df["grade"]       = df["grade"].fillna(df["grade"].mode()[0])

    after = df.isnull().sum().sum()
    print(f"[Preprocessing] Missing values — Before: {before}  After: {after}")
    return df


# ─────────────────────────────────────────────
# STEP 3: Statistical Analysis  (Assignment 5)
# ─────────────────────────────────────────────

def print_statistics(df: pd.DataFrame):
    """Prints describe(), mean, median, std, correlation."""
    print("\n── describe() ──────────────────────────────")
    print(df.describe())
    print("\n── mean() ──────────────────────────────────")
    print(df.mean(numeric_only=True))
    print("\n── median() ────────────────────────────────")
    print(df.median(numeric_only=True))
    print("\n── std() ───────────────────────────────────")
    print(df.std(numeric_only=True))
    print("\n── corr() ──────────────────────────────────")
    print(df.corr(numeric_only=True))


# ─────────────────────────────────────────────
# STEP 4: Encoding  (Assignment 5 — both methods kept)
# ─────────────────────────────────────────────

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

def encode_grade(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Applies BOTH encoding methods from Assignment 5:
      - map()          → grade_encoded_map
      - LabelEncoder   → grade_encoded  (used for modelling)
    Returns updated DataFrame and fitted LabelEncoder.
    """
    df = df.copy()

    # Method 1: map()  (Assignment 5 requirement)
    df["grade_encoded_map"] = df["grade"].map(GRADE_MAP)

    # Method 2: LabelEncoder  (used for ML)
    le = LabelEncoder()
    df["grade_encoded"] = le.fit_transform(df["grade"].astype(str))

    print(f"[Encoding] Grade classes: {list(le.classes_)}")
    return df, le


# ─────────────────────────────────────────────
# STEP 5: Risk Score Formula  (Assignment 5 — unchanged)
# ─────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted risk formula from Assignment 5.
    Normalized to 0–100.  Categorized into Low / Medium / High.
    """
    df = df.copy()

    # Weighted raw score
    df["raw_risk_score"] = (
        0.40 * df["loan_amnt"]    +
        0.25 * df["installment"]  +
        0.20 * df["dti"]          +
        0.15 * df["grade_encoded"]
    )

    # Normalize to 0–100
    mn, mx = df["raw_risk_score"].min(), df["raw_risk_score"].max()
    df["risk_score_norm"] = ((df["raw_risk_score"] - mn) / (mx - mn)) * 100

    # Categorize
    df["risk_category"] = df["risk_score_norm"].apply(categorize_risk)

    dist = df["risk_category"].value_counts().to_dict()
    print(f"[Risk Score] Category distribution: {dist}")
    return df


def categorize_risk(score: float) -> str:
    """Thresholds from Assignment 5 — kept exactly as defined."""
    if score < 33:
        return "Low"
    elif score < 66:
        return "Medium"
    else:
        return "High"


# ─────────────────────────────────────────────
# STEP 6: Feature Engineering  (Assignment 5 — all features kept)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all derived features from Assignment 5:
      is_high_loan, is_high_installment, is_high_dti,
      loan_burden, risk_flag
    """
    df = df.copy()

    df["is_high_loan"]        = (df["loan_amnt"]   > df["loan_amnt"].mean()).astype(int)
    df["is_high_installment"] = (df["installment"] > df["installment"].median()).astype(int)
    df["is_high_dti"]         = (df["dti"]         > 20).astype(int)
    df["loan_burden"]         = df["loan_amnt"] / df["installment"]
    df["risk_flag"]           = (
        (df["is_high_loan"] == 1) & (df["is_high_dti"] == 1)
    ).astype(int)

    print(f"[Feature Eng] New features: is_high_loan, is_high_installment, "
          f"is_high_dti, loan_burden, risk_flag")
    return df


# ─────────────────────────────────────────────
# STEP 7: Encode Target + Scale Features
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "loan_amnt", "installment", "dti", "grade_encoded",
    "is_high_loan", "is_high_installment", "is_high_dti",
    "loan_burden", "risk_flag"
]

def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encodes risk_category → risk_label (0=High, 1=Low, 2=Medium by alpha order).
    Returns df with risk_label column and the fitted LabelEncoder.
    """
    le_target = LabelEncoder()
    df["risk_label"] = le_target.fit_transform(df["risk_category"])
    print(f"[Target Encoding] Classes: {list(le_target.classes_)}")
    return df, le_target


def scale_features(X_train, X_test=None) -> tuple:
    """
    Fits MinMaxScaler on X_train.
    Returns (scaler, X_train_scaled, X_test_scaled).
    X_test_scaled is None if X_test is not provided.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test) if X_test is not None else None
    return scaler, X_train_scaled, X_test_scaled


# ─────────────────────────────────────────────
# CONVENIENCE: preprocess a single live input dict
# (used by app.py for API predictions)
# ─────────────────────────────────────────────

def preprocess_input(raw: dict, le_grade: LabelEncoder, scaler: MinMaxScaler) -> np.ndarray:
    """
    Takes a raw JSON input from the API, applies the same pipeline,
    and returns a scaled numpy array ready for model.predict().

    Expected keys: loan_amnt, installment, dti, grade
    """
    loan_amnt   = float(raw["loan_amnt"])
    installment = float(raw["installment"])
    dti         = float(raw["dti"])

    # Encode grade using the saved LabelEncoder
    grade_str     = str(raw["grade"]).upper()
    grade_encoded = int(le_grade.transform([grade_str])[0])

    # Feature engineering (same logic as training)
    is_high_loan        = 1 if loan_amnt > 1347.32 else 0   # training mean
    is_high_installment = 1 if installment > df_medians["installment"] else 0
    is_high_dti         = 1 if dti > 20 else 0
    loan_burden         = loan_amnt / installment if installment != 0 else 0
    risk_flag           = 1 if (is_high_loan == 1 and is_high_dti == 1) else 0

    features = np.array([[
        loan_amnt, installment, dti, grade_encoded,
        is_high_loan, is_high_installment, is_high_dti,
        loan_burden, risk_flag
    ]])

    return scaler.transform(features)


# Training-set medians saved here for app.py use
df_medians = {"installment": 319.0}   # updated during training
