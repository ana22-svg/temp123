"""
train_model.py
--------------
Run ONCE to train and save models.

Execution Order:
  1. Load + ETL (CSV → JSON → DataFrame)
  2. Missing value handling
  3. Statistical analysis
  4. Grade encoding (map + LabelEncoder)
  5. Risk score formula + categorization   ← Assignment 5 formula kept intact
  6. Feature engineering
  7. Train/test split (stratified)
  8. Scale features
  9. Train 3 Regressors  (predict risk_score_norm)  ← Assignment 5 models kept
 10. Train 3 Classifiers (predict risk_category)    ← NEW: added on top
 11. 5-Fold CV on all models
 12. GridSearchCV on best classifier                ← NEW
 13. Confusion matrix + all Assignment 5 plots      ← NEW plot added
 14. Save models + scalers + encoders with joblib
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.metrics         import (r2_score, mean_absolute_error,
                                     classification_report, confusion_matrix,
                                     accuracy_score)

from preprocess import (load_data, handle_missing, print_statistics,
                        encode_grade, compute_risk_score, categorize_risk,
                        engineer_features, encode_target,
                        scale_features, FEATURE_COLS)

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("model",  exist_ok=True)
os.makedirs("plots",  exist_ok=True)

print("=" * 60)
print("  LOAN RISK PREDICTION — TRAINING PIPELINE")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & ETL
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 1] Loading Data (CSV → JSON → DataFrame)")
df = load_data("data/data.csv")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: MISSING VALUE HANDLING
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 2] Handling Missing Values")
df = handle_missing(df)


# ═══════════════════════════════════════════════════════════════
# PHASE 3: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 3] Statistical Analysis")
print_statistics(df)


# ═══════════════════════════════════════════════════════════════
# PHASE 4: ENCODING
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 4] Encoding Grade Column")
df, le_grade = encode_grade(df)


# ═══════════════════════════════════════════════════════════════
# PHASE 5: RISK SCORE (Assignment 5 formula — unchanged)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 5] Computing Risk Score")
df = compute_risk_score(df)


# ═══════════════════════════════════════════════════════════════
# PHASE 6: FEATURE ENGINEERING (Assignment 5 — all features kept)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 6] Feature Engineering")
df = engineer_features(df)

# Save training medians for live prediction in app.py
training_stats = {
    "loan_amnt_mean":      float(df["loan_amnt"].mean()),
    "installment_median":  float(df["installment"].median()),
    "dti_high_threshold":  20.0
}
with open("model/training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=2)
print(f"[Phase 6] Saved training stats → model/training_stats.json")


# ═══════════════════════════════════════════════════════════════
# PHASE 7: ENCODE TARGET + PREPARE FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 7] Encoding Target + Preparing Feature Matrix")
df, le_target = encode_target(df)

X = df[FEATURE_COLS]
y_reg = df["risk_score_norm"]    # regression target  (0–100 score)
y_clf = df["risk_label"]         # classification target (0/1/2)

print(f"[Phase 7] Features used: {FEATURE_COLS}")
print(f"[Phase 7] X shape: {X.shape}")


# ═══════════════════════════════════════════════════════════════
# PHASE 8: TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 8] Train / Test Split (80/20, stratified on risk_label)")

# Regression split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Classification split — STRATIFIED to preserve class balance
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"[Phase 8] Train: {X_train_r.shape[0]} rows  |  Test: {X_test_r.shape[0]} rows")


# ═══════════════════════════════════════════════════════════════
# PHASE 9: FEATURE SCALING
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 9] Scaling Features (MinMaxScaler)")

scaler_reg, X_train_r_sc, X_test_r_sc = scale_features(X_train_r, X_test_r)
scaler_clf, X_train_c_sc, X_test_c_sc = scale_features(X_train_c, X_test_c)
X_all_sc = scaler_clf.transform(X)   # for CV on full dataset


# ═══════════════════════════════════════════════════════════════
# PHASE 10: REGRESSION MODELS  (Assignment 5 — kept exactly)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 10] Training Regression Models (predicting risk_score_norm)")
print("-" * 50)

reg_models = {
    "Linear Regression":  LinearRegression(),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(
                              n_estimators=200, learning_rate=0.05,
                              max_depth=3, random_state=42)
}

reg_results = []
best_reg_model, best_r2 = None, -999

for name, model in reg_models.items():
    model.fit(X_train_r_sc, y_train_r)
    pred = model.predict(X_test_r_sc)

    r2  = r2_score(y_test_r, pred)
    mae = mean_absolute_error(y_test_r, pred)

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_all_sc, y_reg, cv=5, scoring="r2")

    print(f"{name:25s} → R²: {r2:.4f}  MAE: {mae:.4f}  "
          f"CV Mean: {cv_scores.mean():.4f}  CV Std: {cv_scores.std():.4f}")
    reg_results.append({"Model": name, "R2": r2, "MAE": mae,
                        "CV_Mean": cv_scores.mean(), "CV_Std": cv_scores.std()})

    if r2 > best_r2:
        best_r2        = r2
        best_reg_model = model
        best_reg_name  = name

print(f"\n✓ Best Regressor: {best_reg_name} (R²={best_r2:.4f})")


# ═══════════════════════════════════════════════════════════════
# PHASE 11: CLASSIFICATION MODELS  (NEW — added on top)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 11] Training Classification Models (predicting risk_category)")
print("-" * 50)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(
                               n_estimators=200, learning_rate=0.05,
                               max_depth=3, random_state=42)
}

clf_results = []
best_clf_model, best_acc = None, 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in clf_models.items():
    model.fit(X_train_c_sc, y_train_c)
    preds = model.predict(X_test_c_sc)

    acc      = accuracy_score(y_test_c, preds)
    cv_scores = cross_val_score(model, X_all_sc, y_clf,
                                cv=skf, scoring="accuracy")

    print(f"{name:25s} → Accuracy: {acc:.4f}  "
          f"CV Mean: {cv_scores.mean():.4f}  CV Std: {cv_scores.std():.4f}")
    print(classification_report(y_test_c, preds,
                                target_names=le_target.classes_, zero_division=0))

    clf_results.append({"Model": name, "Accuracy": acc,
                        "CV_Mean": cv_scores.mean(), "CV_Std": cv_scores.std()})

    if acc > best_acc:
        best_acc       = acc
        best_clf_model = model
        best_clf_name  = name

print(f"\n✓ Best Classifier: {best_clf_name} (Accuracy={best_acc:.4f})")


# ═══════════════════════════════════════════════════════════════
# PHASE 12: HYPERPARAMETER TUNING (GridSearchCV on best classifier)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 12] GridSearchCV Hyperparameter Tuning")
print("-" * 50)

param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_c_sc, y_train_c)

print(f"Best Params:   {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

tuned_model = grid_search.best_estimator_
tuned_acc   = accuracy_score(y_test_c, tuned_model.predict(X_test_c_sc))
print(f"Tuned Test Accuracy: {tuned_acc:.4f}")

# Use tuned model if it beats the previous best
if tuned_acc >= best_acc:
    best_clf_model = tuned_model
    best_clf_name  = "Random Forest (Tuned)"
    print("✓ Using tuned model as final classifier")


# ═══════════════════════════════════════════════════════════════
# PHASE 13: VISUALIZATIONS  (All Assignment 5 plots + new ones)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 13] Generating Visualizations → plots/")

# ── Plot 1: Loan Amount Distribution  (Assignment 5)
plt.figure(figsize=(8, 4))
sns.histplot(df["loan_amnt"], bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.tight_layout()
plt.savefig("plots/1_loan_amount_dist.png")
plt.close()

# ── Plot 2: Correlation Heatmap  (Assignment 5)
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/2_correlation_heatmap.png")
plt.close()

# ── Plot 3: Regressor Comparison — R²  (Assignment 5)
res_df = pd.DataFrame(reg_results)
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="R2", data=res_df)
plt.title("Regression Model Comparison (R²)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/3_regressor_r2_comparison.png")
plt.close()

# ── Plot 4: Actual vs Predicted — best regressor  (Assignment 5)
best_reg_pred = best_reg_model.predict(X_test_r_sc)
plt.figure(figsize=(7, 5))
sns.regplot(x=y_test_r, y=best_reg_pred, scatter_kws={"alpha": 0.3})
plt.title(f"Actual vs Predicted Risk Score ({best_reg_name})")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("plots/4_actual_vs_predicted.png")
plt.close()

# ── Plot 5: Cross-Validation Comparison — regressors  (Assignment 5)
cv_df = pd.DataFrame(reg_results)
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="CV_Mean", data=cv_df)
plt.title("Regressor 5-Fold CV Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/5_regressor_cv_comparison.png")
plt.close()

# ── Plot 6: Feature Importance — Random Forest regressor  (Assignment 5)
rf_reg = reg_models["Random Forest"]
plt.figure(figsize=(8, 5))
sns.barplot(x=rf_reg.feature_importances_, y=FEATURE_COLS)
plt.title("Feature Importance — Random Forest Regressor")
plt.tight_layout()
plt.savefig("plots/6_feature_importance_regressor.png")
plt.close()

# ── Plot 7: Risk Category Distribution  (Assignment 5)
plt.figure(figsize=(6, 4))
sns.countplot(x="risk_category", data=df, order=["Low", "Medium", "High"])
plt.title("Risk Category Distribution")
plt.tight_layout()
plt.savefig("plots/7_risk_category_distribution.png")
plt.close()

# ── Plot 8: Classifier Accuracy Comparison  (NEW)
clf_df = pd.DataFrame(clf_results)
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="Accuracy", data=clf_df)
plt.title("Classifier Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/8_classifier_accuracy.png")
plt.close()

# ── Plot 9: Confusion Matrix — best classifier  (NEW)
cm = confusion_matrix(y_test_c, best_clf_model.predict(X_test_c_sc))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title(f"Confusion Matrix — {best_clf_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/9_confusion_matrix.png")
plt.close()

# ── Plot 10: Feature Importance — best classifier RF  (NEW)
rf_clf = clf_models["Random Forest"]
plt.figure(figsize=(8, 5))
sns.barplot(x=rf_clf.feature_importances_, y=FEATURE_COLS)
plt.title("Feature Importance — Random Forest Classifier")
plt.tight_layout()
plt.savefig("plots/10_feature_importance_classifier.png")
plt.close()

print("[Phase 13] ✓ All 10 plots saved to plots/")


# ═══════════════════════════════════════════════════════════════
# PHASE 14: SAVE MODELS + ARTIFACTS
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 14] Saving Models and Artifacts → model/")

joblib.dump(best_reg_model, "model/regressor.pkl")
joblib.dump(best_clf_model, "model/classifier.pkl")
joblib.dump(scaler_clf,     "model/scaler.pkl")
joblib.dump(le_grade,       "model/le_grade.pkl")
joblib.dump(le_target,      "model/le_target.pkl")

# Also save the tuned grid search object
joblib.dump(grid_search,    "model/grid_search.pkl")

print("[Phase 14] ✓ Saved:")
print("  model/regressor.pkl  — best regression model")
print("  model/classifier.pkl — best classification model (tuned RF)")
print("  model/scaler.pkl     — MinMaxScaler")
print("  model/le_grade.pkl   — LabelEncoder for grade")
print("  model/le_target.pkl  — LabelEncoder for risk_category")
print("  model/grid_search.pkl— full GridSearchCV object")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Best Regressor:  {best_reg_name}  (R²={best_r2:.4f})")
print(f"  Best Classifier: {best_clf_name}  (Accuracy={best_acc:.4f})")
print(f"\n  Plots saved: plots/")
print(f"  Models saved: model/")
print(f"\n  Next step → run:  python app.py")
print("=" * 60)
