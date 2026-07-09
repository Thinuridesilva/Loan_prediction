# 🏦 Loan Default Prediction

A machine learning pipeline that predicts whether a borrower will default on a loan, using XGBoost and Random Forest with SHAP explainability and MLflow experiment tracking.

---

## 📌 Problem Statement

Financial institutions lose billions annually to loan defaults. This project builds a binary classifier to identify high-risk borrowers **before** loan approval, using demographic and financial features. The model provides both predictions and human-readable explanations via SHAP values — making it suitable for real-world credit risk teams.

---

## 📊 Dataset

**Source:** [Loan Default Dataset — Kaggle (nikhil1e9)](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

| Property | Value |
|---|---|
| Rows | 255,347 |
| Features | 18 |
| Target | `Default` (0 = repaid, 1 = defaulted) |
| Class imbalance | ~88% No Default / ~12% Default |

**Key features:**

| Feature | Description |
|---|---|
| `Age` | Borrower's age |
| `Income` | Annual income |
| `LoanAmount` | Amount borrowed |
| `CreditScore` | FICO-style credit score |
| `MonthsEmployed` | Employment duration |
| `DTIRatio` | Debt-to-income ratio |
| `InterestRate` | Loan interest rate (%) |
| `LoanTerm` | Repayment term in months |
| `EmploymentType` | Full-time / Part-time / Self-employed / Unemployed |
| `LoanPurpose` | Home / Auto / Education / Business / Other |

> ⚠️ The dataset CSV is not included in this repo due to size. Download it directly from the Kaggle link above and place it as `Loan_default.csv` in the project root.

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-blue)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)
![MLflow](https://img.shields.io/badge/MLflow-tracking-lightblue)

- **Data:** `pandas`, `numpy`
- **Modeling:** `scikit-learn`, `xgboost`
- **Imbalance:** `imbalanced-learn` (SMOTE)
- **Explainability:** `shap`
- **Tracking:** `mlflow`
- **Visualization:** `matplotlib`, `seaborn`

---

## 🗂️ Project Structure

```
loan-default-prediction/
│
├── src/
│   └── loan_default_pipeline.py   # Full pipeline (EDA → training → SHAP → MLflow)
│
├── notebooks/
│   └── 01_eda_modeling.ipynb      # Jupyter/Colab notebook version
│
├── outputs/
│   ├── feature_importance.csv     # SHAP-ranked feature importance
│   └── plots/                     # All generated plots (PNG)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Pipeline Overview

```
Raw CSV
   │
   ▼
Step 1 — Load & Inspect          (shape, dtypes, missing values)
   │
   ▼
Step 2 — EDA                     (distributions, correlation, default rates by category)
   │
   ▼
Step 3 — Feature Engineering     (6 new ratio features, encoding, SMOTE)
   │
   ▼
Step 4 — Train Models            (XGBoost + Random Forest)
   │
   ▼
Step 5 — SHAP Explainability     (beeswarm, waterfall, dependence plots)
   │
   ▼
Step 6 — MLflow Tracking         (log params, metrics, artifacts)
   │
   ▼
Step 7 — Save Models             (.pkl + feature_importance.csv)
```

---

## ⚙️ Feature Engineering

Six new features were derived from existing columns:

| New Feature | Formula | Intuition |
|---|---|---|
| `LoanToIncome` | LoanAmount / Income | Affordability ratio |
| `LoanAmountPerMonth` | LoanAmount / LoanTerm | Monthly burden |
| `IncomePerMonth` | Income / 12 | Normalized income |
| `DebtLoad` | DTIRatio × Income | Absolute debt amount |
| `CreditUtilization` | LoanAmount / CreditScore | Risk-adjusted borrowing |
| `EmployedYears` | MonthsEmployed / 12 | Stability metric |

Class imbalance (88/12 split) was handled with **SMOTE** on the training set only.

---

## 📈 Results

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.7349 | 0.39 | 0.19 | 0.25 |
| Random Forest | 0.7075 | 0.26 | 0.32 | 0.29 |

---

**Top 5 features by SHAP value:**

1. `InterestRate` — higher rate = much higher default risk
2. `DTIRatio` — debt burden is the strongest financial signal
3. `LoanToIncome` — derived feature showing affordability
4. `CreditScore` — lower score strongly associated with default
5. `MonthsEmployed` — employment stability matters

---

## 🖼️ SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to explain both global model behavior and individual predictions.

**Global feature importance (beeswarm):**

![SHAP Beeswarm](outputs/plot_08_shap_beeswarm.png)

**Single prediction waterfall (for a defaulter):**

> Each bar shows how much a feature pushed the prediction toward default (red) or away from it (blue).

---

## 🛠️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Thinuridesilva/Loan_prediction.git
cd Loan_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `Loan_default.csv` from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and place it in the project root.

### 4. Run the pipeline
```bash
python src/loan_default_pipeline.py
```

### 5. View MLflow experiments
```bash
mlflow ui
# Open http://localhost:5000

