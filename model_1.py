
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_recall_curve, auc
)

df = pd.read_csv("Dataset_processed.csv")
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
 
X = df.drop("SepsisLabel", axis=1)
y = df["SepsisLabel"]
groups = df["Patient_ID"]
 
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
 
X_train = X.iloc[train_idx].drop("Patient_ID", axis=1)
X_test  = X.iloc[test_idx].drop("Patient_ID", axis=1)
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]
 
ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Class imbalance ratio: {ratio:.1f}x")
 
print("Training XGBoost model...")
model = XGBClassifier(
    subsample=1.0,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)
model.fit(X_train, y_train)
 
print("Computing evaluation metrics...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
 
roc_auc = roc_auc_score(y_test, y_prob)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
cm = confusion_matrix(y_test, y_pred)
 
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)
 
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"CM      :\n{cm}")
 
artifacts = {
    "model":        model,
    "feature_cols": list(X_train.columns),
    "roc_auc":      roc_auc,
    "pr_auc":       pr_auc,
    "cm":           cm,
    "precision":    precision,
    "recall":       recall,
    "thresholds":   thresholds,
    "y_test":       y_test.values,
    "y_prob":       y_prob,
    "y_pred":       y_pred,
    "importance_df": importance_df,
    "ratio":        ratio,
}
 
joblib.dump(artifacts, "model_artifacts.pkl", compress=3)
print("\n✅  Saved → model_artifacts.pkl")
print("    Copy this file next to app.py before deploying.")
 