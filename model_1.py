import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    GroupKFold
)
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,confusion_matrix,precision_recall_curve,auc

df = pd.read_csv('Dataset_processed.csv')
df = df.drop(columns=['Unnamed: 0'])
X = df.drop('SepsisLabel', axis=1)
y = df['SepsisLabel']

groups = df['Patient_ID']
gss = GroupShuffleSplit(
    test_size=0.2,
    random_state=42
)
train_idx, test_idx = next(
    gss.split(X, y, groups)
)

X_train = X.iloc[train_idx].drop("Patient_ID", axis=1)
X_test = X.iloc[test_idx].drop("Patient_ID", axis=1)

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print(y_train.value_counts())

print(
    y_train.value_counts(normalize=True)
)
ratio = (
    len(y_train[y_train == 0]) /
    len(y_train[y_train == 1])
)

print(ratio)
best_model = XGBClassifier(
    subsample=1.0,
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist'
)


# param_dist = {
#     'n_estimators': [100, 150, 200],
#     'max_depth': [4, 6, 8],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

# group_kfold = GroupKFold(n_splits=3)

# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_dist,
#     n_iter=10,
#     scoring='roc_auc',
#     cv=group_kfold,
#     verbose=2,
#     random_state=42,
#     n_jobs=1
# )
# random_search.fit(
#     X_train,
#     y_train,
#     groups=groups.iloc[train_idx]
# )
# print(random_search.best_params_)

# best_model = random_search.best_estimator_
best_model.fit(X_train,y_train)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print(cm)
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("ROC-AUC:", roc_auc)
from sklearn.metrics import (
    precision_recall_curve,
    auc
)

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = auc(recall, precision)

print("PR-AUC:", pr_auc)
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print(
    importance_df.head(15)
)
custom_threshold = 0.3

y_pred_custom = (
    y_prob >= custom_threshold
).astype(int)

print(
    classification_report(
        y_test,
        y_pred_custom
    )
)