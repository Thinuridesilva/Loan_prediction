!pip install xgboost shap mlflow imbalanced-learn -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/content/drive/MyDrive/Loan_default.csv")

print("=" * 55)
print("STEP 1 — DATA OVERVIEW")
print("=" * 55)
print(f"Shape            : {df.shape}")
print(f"Target balance   :\n{df['Default'].value_counts()}")
print(f"\nMissing values   :\n{df.isnull().sum()}")
print(f"\nDtypes:\n{df.dtypes}")
print(df.head(3))

print("\n" + "=" * 55)
print("STEP 2 — EDA")
print("=" * 55)

# 2a. Target distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Default'].value_counts()
bars = ax.bar(['No Default (0)', 'Default (1)'],
              counts.values,
              color=['#3B8BD4', '#E24B4A'],
              edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 500, f'{val:,}',
            ha='center', fontsize=11, fontweight='bold')
ax.set_title("Target Distribution — Default vs No Default", fontsize=13)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plot_01_target_distribution.png", dpi=150)
plt.show()

# 2b. Numeric feature distributions
num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore',
            'MonthsEmployed', 'NumCreditLines', 'InterestRate',
            'LoanTerm', 'DTIRatio']

df[num_cols].hist(bins=40, figsize=(16, 10),
                  color='#3B8BD4', edgecolor='white')
plt.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot_02_numeric_distributions.png", dpi=150)
plt.show()

# 2c. Correlation heatmap (numeric + target)
plt.figure(figsize=(12, 8))
corr = df[num_cols + ['Default']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask,
            linewidths=0.5, linecolor='white')
plt.title("Correlation Heatmap", fontsize=13)
plt.tight_layout()
plt.savefig("plot_03_correlation_heatmap.png", dpi=150)
plt.show()

# 2d. Default rate by categorical features
cat_cols = ['Education', 'EmploymentType', 'MaritalStatus',
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    rate = df.groupby(col)['Default'].mean().sort_values(ascending=False)
    rate.plot(kind='bar', ax=axes[i], color='#1D9E75', edgecolor='white')
    axes[i].set_title(f"Default Rate by {col}", fontsize=11)
    axes[i].set_ylabel("Default Rate")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=30)
    for p in axes[i].patches:
        axes[i].annotate(f"{p.get_height():.2f}",
                         (p.get_x() + p.get_width()/2, p.get_height()),
                         ha='center', va='bottom', fontsize=8)
# hide unused subplot
axes[-1].set_visible(False)
axes[-2].set_visible(False)
plt.suptitle("Default Rate by Categorical Features", fontsize=14)
plt.tight_layout()
plt.savefig("plot_04_default_rate_categoricals.png", dpi=150)
plt.show()

# 2e. Boxplots: numeric features vs Default
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    df.boxplot(column=col, by='Default', ax=axes[i],
               boxprops=dict(color='#3B8BD4'),
               medianprops=dict(color='#E24B4A', linewidth=2),
               whiskerprops=dict(color='#3B8BD4'),
               capprops=dict(color='#3B8BD4'),
               flierprops=dict(marker='o', markersize=2, alpha=0.3))
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel("Default")
    axes[i].set_ylabel(col)
plt.suptitle("Numeric Features vs Default", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot_05_boxplots.png", dpi=150)
plt.show()

print("\n" + "=" * 55)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 55)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

df_model = df.copy()

# Drop identifier column
df_model.drop(columns=['LoanID'], inplace=True)

# ── 3a. New ratio / interaction features ──
df_model['LoanToIncome']        = df_model['LoanAmount'] / (df_model['Income'] + 1)
df_model['LoanAmountPerMonth']  = df_model['LoanAmount'] / (df_model['LoanTerm'] + 1)
df_model['IncomePerMonth']      = df_model['Income'] / 12
df_model['DebtLoad']            = df_model['DTIRatio'] * df_model['Income']
df_model['CreditUtilization']   = df_model['LoanAmount'] / (df_model['CreditScore'] + 1)
df_model['EmployedYears']       = df_model['MonthsEmployed'] / 12

print("New features added:")
new_feats = ['LoanToIncome', 'LoanAmountPerMonth', 'IncomePerMonth',
             'DebtLoad', 'CreditUtilization', 'EmployedYears']
print(df_model[new_feats].describe().round(2))

# ── 3b. Encode binary Yes/No columns ──
binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_cols:
    df_model[col] = df_model[col].map({'Yes': 1, 'No': 0})

# ── 3c. Ordinal encode Education ──
edu_map = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
df_model['Education'] = df_model['Education'].map(edu_map)

# ── 3d. One-hot encode remaining categoricals ──
df_model = pd.get_dummies(
    df_model,
    columns=['EmploymentType', 'MaritalStatus', 'LoanPurpose'],
    drop_first=True
)

print(f"\nFinal shape after encoding: {df_model.shape}")
print(f"Columns: {list(df_model.columns)}")

# ── 3e. Train / Test split ──
X = df_model.drop(columns=['Default'])
y = df_model['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.4f}")
print(f"Test  default rate: {y_test.mean():.4f}")

# ── 3f. SMOTE to fix class imbalance ──
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"\nBefore SMOTE — 0: {(y_train==0).sum():,} | 1: {(y_train==1).sum():,}")
print(f"After  SMOTE — 0: {(y_train_res==0).sum():,} | 1: {(y_train_res==1).sum():,}")

print("\n" + "=" * 55)
print("STEP 4 — MODEL TRAINING")
print("=" * 55)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)

# ── 4a. XGBoost ──
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,          # SMOTE already balanced
    eval_metric='logloss',
    early_stopping_rounds=25,    # moved here in XGBoost >= 1.6
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print(f"  Best iteration: {xgb_model.best_iteration}")

# ── 4b. Random Forest ──
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)
print("  Done.")

# ── 4c. Evaluate both models ──
print("\n" + "-" * 55)
results = {}
for name, model in [("XGBoost", xgb_model), ("Random Forest", rf_model)]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ap    = average_precision_score(y_test, proba)
    results[name] = {'preds': preds, 'proba': proba, 'auc': auc, 'ap': ap}
    print(f"\n{'='*10} {name} {'='*10}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Avg Prec : {ap:.4f}")
    print(classification_report(y_test, preds, target_names=['No Default','Default']))

# ── 4d. ROC curve comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, model in [("XGBoost", xgb_model), ("Random Forest", rf_model)]:
    RocCurveDisplay.from_estimator(model, X_test, y_test,
                                   ax=ax, name=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',linewidth=0.8)
ax.set_title("ROC Curve Comparison", fontsize=12)

ax = axes[1]
for name, model in [("XGBoost", xgb_model), ("Random Forest", rf_model)]:
    precision, recall, _ = precision_recall_curve(
        y_test, model.predict_proba(X_test)[:, 1])
    ax.plot(recall, precision, label=f"{name} (AP={results[name]['ap']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig("plot_06_roc_pr_curves.png", dpi=150)
plt.show()

# ── 4e. Confusion matrices ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, model) in zip(axes, [("XGBoost", xgb_model), ("Random Forest", rf_model)]):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['No Default', 'Default'],
        cmap='Blues', ax=ax
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=12)
plt.tight_layout()
plt.savefig("plot_07_confusion_matrices.png", dpi=150)
plt.show()

print("\n" + "=" * 55)
print("STEP 5 — SHAP EXPLAINABILITY")
print("=" * 55)

import shap

# Use a 2,000-row sample for speed
X_shap = X_test.sample(n=2000, random_state=42)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)

# 5a. Beeswarm summary (global feature importance with direction)
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="dot",
                  max_display=15, show=False)
plt.title("SHAP Beeswarm — Top 15 Features (XGBoost)", fontsize=12)
plt.tight_layout()
plt.savefig("plot_08_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()

# 5b. Bar chart of mean |SHAP| values
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar",
                  max_display=15, show=False)
plt.title("SHAP Feature Importance — Mean |SHAP| (XGBoost)", fontsize=12)
plt.tight_layout()
plt.savefig("plot_09_shap_importance_bar.png", dpi=150, bbox_inches='tight')
plt.show()

# 5c. Waterfall plot — single prediction (first defaulter in test set)
defaulter_idx = X_shap[y_test.loc[X_shap.index] == 1].index[0]
sample_pos    = X_shap.index.get_loc(defaulter_idx)

shap_exp = shap.Explanation(
    values      = shap_values[sample_pos],
    base_values = explainer.expected_value,
    data        = X_shap.iloc[sample_pos].values,
    feature_names = X_shap.columns.tolist()
)
plt.figure()
shap.plots.waterfall(shap_exp, max_display=15, show=False)
plt.title(f"SHAP Waterfall — Sample Defaulter (LoanID idx {defaulter_idx})", fontsize=11)
plt.tight_layout()
plt.savefig("plot_10_shap_waterfall_defaulter.png", dpi=150, bbox_inches='tight')
plt.show()

# 5d. SHAP dependence plot — top 2 features
top_features = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_shap.columns
).nlargest(2).index.tolist()

for feat in top_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X_shap, show=False)
    plt.title(f"SHAP Dependence — {feat}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plot_11_shap_dependence_{feat}.png", dpi=150, bbox_inches='tight')
    plt.show()

 print("\n" + "=" * 55)
print("STEP 6 — MLFLOW TRACKING")
print("=" * 55)

import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

mlflow.set_experiment("loan-default-prediction")

# Log XGBoost run
with mlflow.start_run(run_name="xgboost_smote_v1"):
    params = {
        "model"              : "XGBoostClassifier",
        "n_estimators"       : xgb_model.best_iteration,
        "max_depth"          : 6,
        "learning_rate"      : 0.05,
        "subsample"          : 0.8,
        "colsample_bytree"   : 0.8,
        "smote"              : True,
        "train_size"         : X_train.shape[0],
        "test_size"          : X_test.shape[0],
        "n_features"         : X_train.shape[1],
    }
    mlflow.log_params(params)

    preds = xgb_model.predict(X_test)
    proba = xgb_model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_roc"          : roc_auc_score(y_test, proba),
        "avg_precision"    : average_precision_score(y_test, proba),
        "precision"        : precision_score(y_test, preds),
        "recall"           : recall_score(y_test, preds),
        "f1"               : f1_score(y_test, preds),
    }
    mlflow.log_metrics(metrics)

    # Log model artifact
    mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model")

    # Log SHAP plot as artifact
    mlflow.log_artifact("plot_08_shap_beeswarm.png")
    mlflow.log_artifact("plot_06_roc_pr_curves.png")

    print(f"\nXGBoost run logged:")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\nRun: mlflow ui   →   http://localhost:5000")

# Log Random Forest run
with mlflow.start_run(run_name="random_forest_smote_v1"):
    mlflow.log_params({
        "model"        : "RandomForestClassifier",
        "n_estimators" : 200,
        "max_depth"    : 12,
        "smote"        : True,
    })
    preds = rf_model.predict(X_test)
    proba = rf_model.predict_proba(X_test)[:, 1]
    mlflow.log_metrics({
        "auc_roc"       : roc_auc_score(y_test, proba),
        "avg_precision" : average_precision_score(y_test, proba),
        "precision"     : precision_score(y_test, preds),
        "recall"        : recall_score(y_test, preds),
        "f1"            : f1_score(y_test, preds),
    })
    mlflow.sklearn.log_model(rf_model, artifact_path="rf_model")
    print("\nRandom Forest run logged.")

   print("\n" + "=" * 55)
print("STEP 7 — SAVE MODEL & FEATURE IMPORTANCE")
print("=" * 55)

import joblib

# Save models
joblib.dump(xgb_model, "xgb_loan_default_model.pkl")
joblib.dump(rf_model,  "rf_loan_default_model.pkl")
print("Models saved: xgb_loan_default_model.pkl, rf_loan_default_model.pkl")

# Feature importance table from SHAP
shap_importance = pd.DataFrame({
    'Feature'     : X_shap.columns,
    'Mean_SHAP'   : np.abs(shap_values).mean(axis=0),
    'XGB_Gain'    : xgb_model.get_booster().get_score(importance_type='gain').values()
                    if len(xgb_model.get_booster().get_score(importance_type='gain')) == len(X_shap.columns)
                    else np.nan
}).sort_values('Mean_SHAP', ascending=False)

print("\nTop 10 Features (by SHAP):")
print(shap_importance.head(10).to_string(index=False))
shap_importance.to_csv("feature_importance.csv", index=False)
print("\nSaved: feature_importance.csv")

print("\n" + "=" * 55)
print("PIPELINE COMPLETE!")
print("=" * 55)
print("""
Output files:
  plot_01_target_distribution.png
  plot_02_numeric_distributions.png
  plot_03_correlation_heatmap.png
  plot_04_default_rate_categoricals.png
  plot_05_boxplots.png
  plot_06_roc_pr_curves.png
  plot_07_confusion_matrices.png
  plot_08_shap_beeswarm.png
  plot_09_shap_importance_bar.png
  plot_10_shap_waterfall_defaulter.png
  plot_11_shap_dependence_<feature>.png
  xgb_loan_default_model.pkl
  rf_loan_default_model.pkl
  feature_importance.csv

MLflow UI:  run  'mlflow ui'  in terminal
""")
