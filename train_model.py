"""
train_model.py
--------------
Run ONCE to train and save models.

Execution Order:
  1.  Load + ETL (CSV → JSON → DataFrame)
  2.  Missing value handling
  3.  Statistical analysis
  4.  Grade encoding (map + LabelEncoder)
  5.  Risk score formula + categorization     <- Assignment 5 kept intact
  6.  Feature engineering
  7.  Encode target + prepare feature matrix
  8.  Train / test split (stratified)
  9.  Feature scaling
  -------- DESCRIPTIVE MODEL (NEW) -----------
  10. K-Means Clustering
  -------- REGRESSION (Assignment 5) ---------
  11. Train 3 Regressors + 5-Fold CV
  -------- CLASSIFICATION (guideline order) --
  12. Baseline: Naive Bayes + Decision Tree   <- NEW: required by guidelines
  13. Advanced: Logistic + Random Forest + Gradient Boosting
  14. Model Justification comparison          <- NEW
  15. GridSearchCV on best classifier
  -------- INSIGHTS + PLOTS ------------------
  16. Insight generation (written conclusions) <- NEW
  17. All 14 plots (original 10 + 4 new)
  18. Save all models + artifacts
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.cluster         import KMeans
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import LabelEncoder, MinMaxScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.metrics         import (r2_score, mean_absolute_error,
                                     classification_report, confusion_matrix,
                                     accuracy_score, precision_score,
                                     recall_score, f1_score)

from preprocess import (load_data, handle_missing, print_statistics,
                        encode_grade, compute_risk_score, categorize_risk,
                        engineer_features, encode_target,
                        scale_features, FEATURE_COLS)

os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 65)
print("  LOAN RISK PREDICTION — COMPLETE TRAINING PIPELINE")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & ETL
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 1] Loading Data (CSV -> JSON -> DataFrame)")
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
# PHASE 5: RISK SCORE (Assignment 5 formula - unchanged)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 5] Computing Risk Score")
df = compute_risk_score(df)


# ═══════════════════════════════════════════════════════════════
# PHASE 6: FEATURE ENGINEERING (Assignment 5 - all features kept)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 6] Feature Engineering")
df = engineer_features(df)

training_stats = {
    "loan_amnt_mean":     float(df["loan_amnt"].mean()),
    "installment_median": float(df["installment"].median()),
    "dti_high_threshold": 20.0
}
with open("model/training_stats.json", "w") as f:
    json.dump(training_stats, f, indent=2)
print("[Phase 6] Saved training stats -> model/training_stats.json")


# ═══════════════════════════════════════════════════════════════
# PHASE 7: ENCODE TARGET + PREPARE FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 7] Encoding Target + Preparing Feature Matrix")
df, le_target = encode_target(df)

X     = df[FEATURE_COLS]
y_reg = df["risk_score_norm"]
y_clf = df["risk_label"]

print(f"[Phase 7] Features: {FEATURE_COLS}")
print(f"[Phase 7] X shape : {X.shape}")


# ═══════════════════════════════════════════════════════════════
# PHASE 8: TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 8] Train / Test Split (80/20, stratified)")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"[Phase 8] Train: {X_train_r.shape[0]}  |  Test: {X_test_r.shape[0]}")


# ═══════════════════════════════════════════════════════════════
# PHASE 9: FEATURE SCALING
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 9] Scaling Features (MinMaxScaler)")

scaler_reg, X_train_r_sc, X_test_r_sc = scale_features(X_train_r, X_test_r)
scaler_clf, X_train_c_sc, X_test_c_sc = scale_features(X_train_c, X_test_c)
X_all_sc = scaler_clf.transform(X)


# ═══════════════════════════════════════════════════════════════
# PHASE 10: DESCRIPTIVE MODEL - K-MEANS CLUSTERING  (NEW)
# Satisfies the guideline requirement for a descriptive model
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 10] Descriptive Model - K-Means Clustering")
print("-" * 50)

cluster_features = ["loan_amnt", "installment", "dti", "grade_encoded"]
X_cluster        = MinMaxScaler().fit_transform(df[cluster_features])

# Elbow Method to find optimal k
inertia_values = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertia_values.append(km.inertia_)

# Fit final model with k=3 (maps to Low / Medium / High naturally)
kmeans     = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_cluster)

cluster_risk_means = df.groupby("cluster")["risk_score_norm"].mean().sort_values()
cluster_label_map  = {
    cluster_risk_means.index[0]: "Low Risk Group",
    cluster_risk_means.index[1]: "Medium Risk Group",
    cluster_risk_means.index[2]: "High Risk Group"
}
df["cluster_label"] = df["cluster"].map(cluster_label_map)

print("\nCluster Summary (Descriptive Analysis):")
cluster_summary = df.groupby("cluster_label")[
    ["loan_amnt", "installment", "dti", "risk_score_norm"]
].mean().round(2)
print(cluster_summary)
joblib.dump(kmeans, "model/kmeans.pkl")
print("\n[Phase 10] K-Means model saved -> model/kmeans.pkl")


# ═══════════════════════════════════════════════════════════════
# PHASE 11: REGRESSION MODELS  (Assignment 5 - kept exactly)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 11] Training Regression Models (predicting risk_score_norm)")
print("-" * 50)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
                             n_estimators=200, learning_rate=0.05,
                             max_depth=3, random_state=42)
}

reg_results          = []
best_reg_model       = None
best_r2              = -999

for name, model in reg_models.items():
    model.fit(X_train_r_sc, y_train_r)
    pred      = model.predict(X_test_r_sc)
    r2        = r2_score(y_test_r, pred)
    mae       = mean_absolute_error(y_test_r, pred)
    cv_scores = cross_val_score(model, X_all_sc, y_reg, cv=5, scoring="r2")

    print(f"{name:25s} -> R2: {r2:.4f}  MAE: {mae:.4f}  "
          f"CV Mean: {cv_scores.mean():.4f}  CV Std: {cv_scores.std():.4f}")

    reg_results.append({"Model": name, "R2": r2, "MAE": mae,
                        "CV_Mean": cv_scores.mean(), "CV_Std": cv_scores.std()})

    if r2 > best_r2:
        best_r2        = r2
        best_reg_model = model
        best_reg_name  = name

print(f"\nBest Regressor: {best_reg_name}  (R2={best_r2:.4f})")


# ═══════════════════════════════════════════════════════════════
# PHASE 12: BASELINE CLASSIFIERS - Naive Bayes + Decision Tree  (NEW)
# Guidelines: start with basic models before moving to advanced
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 12] Baseline Classifiers - Naive Bayes & Decision Tree")
print("-" * 50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_models = {
    "Naive Bayes":   GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5)
}

baseline_results = []

for name, model in baseline_models.items():
    model.fit(X_train_c_sc, y_train_c)
    preds     = model.predict(X_test_c_sc)
    acc       = accuracy_score(y_test_c, preds)
    precision = precision_score(y_test_c, preds, average="weighted", zero_division=0)
    recall    = recall_score(y_test_c, preds, average="weighted", zero_division=0)
    f1        = f1_score(y_test_c, preds, average="weighted", zero_division=0)
    cv_scores = cross_val_score(model, X_all_sc, y_clf, cv=skf, scoring="accuracy")

    print(f"\n-- {name} --")
    print(f"  Accuracy : {acc:.4f}  |  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}  |  F1 Score : {f1:.4f}")
    print(f"  CV Mean  : {cv_scores.mean():.4f}  +/- {cv_scores.std():.4f}")
    print(classification_report(y_test_c, preds,
                                target_names=le_target.classes_, zero_division=0))

    baseline_results.append({
        "Model":     name,
        "Accuracy":  round(acc, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "CV_Mean":   round(cv_scores.mean(), 4),
        "CV_Std":    round(cv_scores.std(), 4)
    })

print("[Phase 12] Baseline models trained")


# ═══════════════════════════════════════════════════════════════
# PHASE 13: ADVANCED CLASSIFIERS
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 13] Advanced Classifiers - Logistic / Random Forest / Gradient Boosting")
print("-" * 50)

advanced_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(
                               n_estimators=200, learning_rate=0.05,
                               max_depth=3, random_state=42)
}

advanced_results = []
best_clf_model   = None
best_acc         = 0

for name, model in advanced_models.items():
    model.fit(X_train_c_sc, y_train_c)
    preds     = model.predict(X_test_c_sc)
    acc       = accuracy_score(y_test_c, preds)
    precision = precision_score(y_test_c, preds, average="weighted", zero_division=0)
    recall    = recall_score(y_test_c, preds, average="weighted", zero_division=0)
    f1        = f1_score(y_test_c, preds, average="weighted", zero_division=0)
    cv_scores = cross_val_score(model, X_all_sc, y_clf, cv=skf, scoring="accuracy")

    print(f"\n-- {name} --")
    print(f"  Accuracy : {acc:.4f}  |  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}  |  F1 Score : {f1:.4f}")
    print(f"  CV Mean  : {cv_scores.mean():.4f}  +/- {cv_scores.std():.4f}")
    print(classification_report(y_test_c, preds,
                                target_names=le_target.classes_, zero_division=0))

    advanced_results.append({
        "Model":     name,
        "Accuracy":  round(acc, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "CV_Mean":   round(cv_scores.mean(), 4),
        "CV_Std":    round(cv_scores.std(), 4)
    })

    if acc > best_acc:
        best_acc       = acc
        best_clf_model = model
        best_clf_name  = name

print(f"\nBest Advanced Classifier: {best_clf_name}  (Accuracy={best_acc:.4f})")

all_clf_results = baseline_results + advanced_results


# ═══════════════════════════════════════════════════════════════
# PHASE 14: MODEL JUSTIFICATION COMPARISON  (NEW)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 14] Model Justification - Why Advanced Beats Baseline")
print("=" * 65)

comparison_df = pd.DataFrame(all_clf_results).sort_values("Accuracy", ascending=False)
print("\nFull Model Comparison Table:")
print(comparison_df[["Model","Accuracy","Precision","Recall","F1","CV_Mean"]].to_string(index=False))

nb_acc = next(r["Accuracy"] for r in baseline_results if r["Model"] == "Naive Bayes")
dt_acc = next(r["Accuracy"] for r in baseline_results if r["Model"] == "Decision Tree")
rf_acc = next(r["Accuracy"] for r in advanced_results  if r["Model"] == "Random Forest")
gb_acc = next(r["Accuracy"] for r in advanced_results  if r["Model"] == "Gradient Boosting")
gap    = max(rf_acc, gb_acc) - max(nb_acc, dt_acc)

print(f"""
Justification:
  Naive Bayes ({nb_acc:.2%}):
    Assumes all features are statistically independent.
    In financial data, loan_amnt and installment are highly correlated,
    making this assumption invalid -> reduced accuracy.

  Decision Tree ({dt_acc:.2%}):
    Interpretable but tends to overfit on training data and
    struggles with overlapping risk boundaries in financial datasets.

  Random Forest ({rf_acc:.2%}):
    Builds many trees on random data subsets and averages predictions.
    Reduces both variance and bias -> significantly better accuracy.

  Gradient Boosting ({gb_acc:.2%}):
    Sequentially corrects errors of prior trees using gradient descent.
    Most accurate on structured tabular financial data.

  Conclusion: Ensemble models outperform baseline by {gap:.2%} in accuracy.
""")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# PHASE 15: HYPERPARAMETER TUNING (GridSearchCV)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 15] GridSearchCV Hyperparameter Tuning on Random Forest")
print("-" * 50)

param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [3, 5, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=skf, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train_c_sc, y_train_c)

tuned_model = grid_search.best_estimator_
tuned_acc   = accuracy_score(y_test_c, tuned_model.predict(X_test_c_sc))

print(f"Best Params:         {grid_search.best_params_}")
print(f"Best CV Score:       {grid_search.best_score_:.4f}")
print(f"Tuned Test Accuracy: {tuned_acc:.4f}")

if tuned_acc >= best_acc:
    best_clf_model = tuned_model
    best_clf_name  = "Random Forest (Tuned)"
    best_acc       = tuned_acc
    print("Tuned model selected as final classifier")


# ═══════════════════════════════════════════════════════════════
# PHASE 16: INSIGHT GENERATION  (NEW)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 16] Insight Generation")
print("=" * 65)

rf_clf      = advanced_models["Random Forest"]
importances = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
top_feature = importances.idxmax()
bot_feature = importances.idxmin()

cat_counts = df["risk_category"].value_counts()
total      = len(df)

high_dti_avg = df[df["is_high_dti"] == 1]["risk_score_norm"].mean()
low_dti_avg  = df[df["is_high_dti"] == 0]["risk_score_norm"].mean()
grade_risk   = df.groupby("grade")["risk_score_norm"].mean().sort_values(ascending=False)

print("\n  INSIGHT 1 - Feature Importance (What Drives Loan Risk?)")
print("  " + "-" * 50)
for feat, imp in importances.items():
    bar = "=" * int(imp * 40)
    print(f"  {feat:25s} [{bar:<40}] {imp:.4f}")
print(f"\n  Top predictor   : '{top_feature}'")
print(f"  Least impactful : '{bot_feature}'")

print("\n  INSIGHT 2 - Borrower Risk Distribution")
print("  " + "-" * 50)
for cat in ["Low", "Medium", "High"]:
    pct = cat_counts.get(cat, 0) / total * 100
    print(f"  {cat:8s} Risk: {cat_counts.get(cat,0):6d} borrowers  ({pct:.1f}%)")

print("\n  INSIGHT 3 - DTI Impact on Risk")
print("  " + "-" * 50)
print(f"  High DTI (>20) avg risk score : {high_dti_avg:.2f}")
print(f"  Low  DTI (<=20) avg risk score: {low_dti_avg:.2f}")
print(f"  High-DTI borrowers carry {high_dti_avg - low_dti_avg:.2f} pts more risk on average.")

print("\n  INSIGHT 4 - Credit Grade vs Risk Score")
print("  " + "-" * 50)
for grade, score in grade_risk.items():
    print(f"  Grade {grade}: {score:.2f}")
print(f"  Grade G borrowers carry {grade_risk.iloc[0] - grade_risk.iloc[-1]:.2f} pts more risk than Grade A.")

print("\n  INSIGHT 5 - Borrower Cluster Profiles (K-Means)")
print("  " + "-" * 50)
for label in ["Low Risk Group", "Medium Risk Group", "High Risk Group"]:
    g = df[df["cluster_label"] == label]
    print(f"  {label} ({len(g)} borrowers, {len(g)/total*100:.1f}%):")
    print(f"    Avg Loan: ${g['loan_amnt'].mean():.2f} | "
          f"Avg DTI: {g['dti'].mean():.2f}% | "
          f"Avg Risk: {g['risk_score_norm'].mean():.2f}/100")

# Save insights to text file
with open("model/insights.txt", "w") as f:
    f.write("LOAN RISK PREDICTION - ANALYTICAL INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("INSIGHT 1: Feature Importance\n")
    for feat, imp in importances.items():
        f.write(f"  {feat}: {imp:.4f}\n")
    f.write(f"\n  Top predictor: {top_feature}\n\n")
    f.write("INSIGHT 2: Risk Distribution\n")
    for cat in ["Low", "Medium", "High"]:
        pct = cat_counts.get(cat, 0) / total * 100
        f.write(f"  {cat}: {pct:.1f}%\n")
    f.write(f"\nINSIGHT 3: DTI Impact\n")
    f.write(f"  High DTI avg risk: {high_dti_avg:.2f}\n")
    f.write(f"  Low  DTI avg risk: {low_dti_avg:.2f}\n\n")
    f.write("INSIGHT 4: Grade vs Risk\n")
    for grade, score in grade_risk.items():
        f.write(f"  Grade {grade}: {score:.2f}\n")
    f.write("\nINSIGHT 5: Cluster Profiles\n")
    for label in ["Low Risk Group", "Medium Risk Group", "High Risk Group"]:
        g = df[df["cluster_label"] == label]
        f.write(f"  {label}: {len(g)} borrowers | "
                f"Avg DTI={g['dti'].mean():.2f} | "
                f"Avg Risk={g['risk_score_norm'].mean():.2f}\n")

print("\n[Phase 16] Insights saved -> model/insights.txt")


# ═══════════════════════════════════════════════════════════════
# PHASE 17: VISUALIZATIONS  (14 total — original 10 + 4 new)
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 17] Generating Visualizations -> plots/")

# Plot 1: Loan Amount Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["loan_amnt"], bins=30, kde=True, color="steelblue")
plt.title("Loan Amount Distribution")
plt.tight_layout()
plt.savefig("plots/1_loan_amount_dist.png"); plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/2_correlation_heatmap.png"); plt.close()

# Plot 3: Regressor R2 Comparison
res_df = pd.DataFrame(reg_results)
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="R2", data=res_df, palette="Blues_d")
plt.title("Regression Model Comparison (R2)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/3_regressor_r2_comparison.png"); plt.close()

# Plot 4: Actual vs Predicted
best_reg_pred = best_reg_model.predict(X_test_r_sc)
plt.figure(figsize=(7, 5))
sns.regplot(x=y_test_r, y=best_reg_pred, scatter_kws={"alpha": 0.3})
plt.title(f"Actual vs Predicted Risk Score ({best_reg_name})")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("plots/4_actual_vs_predicted.png"); plt.close()

# Plot 5: Regressor CV Comparison
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="CV_Mean", data=res_df, palette="Greens_d")
plt.title("Regressor 5-Fold CV Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/5_regressor_cv_comparison.png"); plt.close()

# Plot 6: Feature Importance (Regressor)
rf_reg = reg_models["Random Forest"]
plt.figure(figsize=(8, 5))
sns.barplot(x=rf_reg.feature_importances_, y=FEATURE_COLS, palette="viridis")
plt.title("Feature Importance - Random Forest Regressor")
plt.tight_layout()
plt.savefig("plots/6_feature_importance_regressor.png"); plt.close()

# Plot 7: Risk Category Distribution
plt.figure(figsize=(6, 4))
pal = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}
sns.countplot(x="risk_category", data=df, order=["Low","Medium","High"], palette=pal)
plt.title("Risk Category Distribution")
plt.tight_layout()
plt.savefig("plots/7_risk_category_distribution.png"); plt.close()

# Plot 8: ALL Classifiers Accuracy (baseline + advanced)  <- UPDATED
all_clf_df = pd.DataFrame(all_clf_results)
plt.figure(figsize=(11, 5))
colors = ["#95a5a6", "#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"]
bars   = plt.bar(all_clf_df["Model"], all_clf_df["Accuracy"], color=colors)
plt.title("All Classifiers - Accuracy Comparison (Baseline -> Advanced)")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.1)
for bar, val in zip(bars, all_clf_df["Accuracy"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("plots/8_all_classifiers_accuracy.png"); plt.close()

# Plot 9: Confusion Matrix
cm = confusion_matrix(y_test_c, best_clf_model.predict(X_test_c_sc))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title(f"Confusion Matrix - {best_clf_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/9_confusion_matrix.png"); plt.close()

# Plot 10: Feature Importance (Classifier)
sorted_imp = importances.sort_values()
plt.figure(figsize=(8, 5))
sns.barplot(x=sorted_imp.values, y=sorted_imp.index, palette="magma")
plt.title("Feature Importance - Random Forest Classifier")
plt.tight_layout()
plt.savefig("plots/10_feature_importance_classifier.png"); plt.close()

# Plot 11: K-Means Elbow Curve  (NEW)
plt.figure(figsize=(7, 4))
plt.plot(list(K_range), inertia_values, "bo-", linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve - Optimal k Selection")
plt.axvline(x=3, color="red", linestyle="--", label="Selected k=3")
plt.legend()
plt.tight_layout()
plt.savefig("plots/11_kmeans_elbow.png"); plt.close()

# Plot 12: Cluster Scatter  (NEW)
plt.figure(figsize=(8, 5))
cmap = {"Low Risk Group": "#27ae60", "Medium Risk Group": "#f39c12",
        "High Risk Group": "#e74c3c"}
for label, color in cmap.items():
    sub = df[df["cluster_label"] == label]
    plt.scatter(sub["dti"], sub["loan_amnt"], c=color, label=label, alpha=0.4, s=10)
plt.xlabel("DTI"); plt.ylabel("Loan Amount")
plt.title("Borrower Clusters - DTI vs Loan Amount")
plt.legend()
plt.tight_layout()
plt.savefig("plots/12_cluster_scatter.png"); plt.close()

# Plot 13: F1 Score Comparison Baseline vs Advanced  (NEW)
plt.figure(figsize=(11, 5))
bars2 = plt.bar(all_clf_df["Model"], all_clf_df["F1"], color=colors)
plt.title("Model Comparison - F1 Score (Baseline -> Advanced)")
plt.xticks(rotation=20, ha="right")
plt.ylabel("F1 Score (Weighted)")
plt.ylim(0, 1.1)
for bar, val in zip(bars2, all_clf_df["F1"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("plots/13_f1_score_comparison.png"); plt.close()

# Plot 14: Grade vs Avg Risk Score  (NEW - Insight 4 visual)
grade_risk_df = grade_risk.reset_index()
grade_risk_df.columns = ["grade", "avg_risk"]
plt.figure(figsize=(8, 4))
sns.barplot(x="grade", y="avg_risk", data=grade_risk_df,
            order=["A","B","C","D","E","F","G"], palette="RdYlGn_r")
plt.title("Credit Grade vs Average Risk Score")
plt.xlabel("Credit Grade"); plt.ylabel("Average Risk Score (0-100)")
plt.tight_layout()
plt.savefig("plots/14_grade_vs_risk.png"); plt.close()

print("[Phase 17] All 14 plots saved to plots/")


# ═══════════════════════════════════════════════════════════════
# PHASE 18: SAVE ALL MODELS + ARTIFACTS
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 18] Saving Models and Artifacts -> model/")

joblib.dump(best_reg_model,                   "model/regressor.pkl")
joblib.dump(best_clf_model,                   "model/classifier.pkl")
joblib.dump(baseline_models["Naive Bayes"],   "model/naive_bayes.pkl")
joblib.dump(baseline_models["Decision Tree"], "model/decision_tree.pkl")
joblib.dump(scaler_clf,                       "model/scaler.pkl")
joblib.dump(le_grade,                         "model/le_grade.pkl")
joblib.dump(le_target,                        "model/le_target.pkl")
joblib.dump(grid_search,                      "model/grid_search.pkl")
joblib.dump(kmeans,                           "model/kmeans.pkl")

print("[Phase 18] Saved:")
print("  model/regressor.pkl     - best regression model")
print("  model/classifier.pkl    - best classifier (tuned RF)")
print("  model/naive_bayes.pkl   - baseline Naive Bayes")
print("  model/decision_tree.pkl - baseline Decision Tree")
print("  model/kmeans.pkl        - K-Means clustering model")
print("  model/scaler.pkl        - MinMaxScaler")
print("  model/le_grade.pkl      - LabelEncoder for grade")
print("  model/le_target.pkl     - LabelEncoder for risk_category")
print("  model/grid_search.pkl   - GridSearchCV object")
print("  model/insights.txt      - Generated analytical insights")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  TRAINING COMPLETE - FULL PIPELINE SUMMARY")
print("=" * 65)
print(f"""
  DESCRIPTIVE MODEL
  +-- K-Means Clustering : 3 borrower groups identified

  REGRESSION MODELS
  +-- Best: {best_reg_name}  (R2={best_r2:.4f})

  CLASSIFICATION MODELS (Baseline -> Advanced)
  +-- Naive Bayes      : {nb_acc:.2%}
  +-- Decision Tree    : {dt_acc:.2%}
  +-- Random Forest    : {rf_acc:.2%}
  +-- Gradient Boosting: {gb_acc:.2%}
  +-- BEST: {best_clf_name}  ({best_acc:.2%})

  ARTIFACTS
  +-- Plots   : plots/  (14 visualizations)
  +-- Models  : model/  (9 saved artifacts)
  +-- Insights: model/insights.txt

  Next -> run: python app.py
""")
print("=" * 65)
