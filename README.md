# Loan Default Prediction

A machine learning pipeline that predicts whether a borrower will default on a loan. Compares XGBoost, LightGBM, Random Forest, and Logistic Regression with SHAP explainability, Optuna hyperparameter tuning, business cost threshold optimization, and a FastAPI REST endpoint served via Docker.

## Results

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost (Optuna tuned) | 0.7390 | — | — | 0.21 |
| LightGBM | 0.7354 | 0.39 | 0.18 | 0.25 |
| XGBoost | 0.7349 | 0.39 | 0.19 | 0.25 |
| Random Forest | 0.7075 | 0.26 | 0.32 | 0.29 |
| Logistic Regression | 0.6974 | 0.31 | 0.27 | 0.29 |

The Optuna-tuned XGBoost achieved the best AUC-ROC of 0.7390 after 30 trials.

## Business Cost Optimization

By tuning the decision threshold from 0.50 to 0.06, missed defaulters dropped from 4,831 to 540 (89% reduction), reducing total business cost from $49.1M to $20.5M — saving approximately $28.6M. False alarms increased but at a much lower cost per case ($500 vs $10,000 per missed defaulter).

## Dataset

[Loan Default Dataset — Kaggle (nikhil1e9)](https://www.kaggle.com/datasets/nikhil1e9/loan-default) — 255,347 rows, 18 features, binary target (0 = repaid, 1 = defaulted), 88/12 class imbalance.

The dataset CSV is not included in this repo. Download it from Kaggle and place it as `Loan_default.csv` in the project root.

## Key Findings

- InterestRate and DTIRatio are the strongest predictors of default (SHAP analysis)
- A derived feature LoanToIncome (LoanAmount / Income) ranked in the top 5 features
- SMOTE on training data alone improved recall on the minority class
- Default threshold of 0.5 optimizes accuracy but is costly in a banking context — threshold 0.06 minimizes total financial loss

## Feature Engineering

Six new features were derived from existing columns:

- `LoanToIncome` — LoanAmount / Income
- `LoanAmountPerMonth` — LoanAmount / LoanTerm
- `IncomePerMonth` — Income / 12
- `DebtLoad` — DTIRatio × Income
- `CreditUtilization` — LoanAmount / CreditScore
- `EmployedYears` — MonthsEmployed / 12

## Tech Stack

- Data: pandas, numpy
- Modeling: scikit-learn, xgboost, lightgbm
- Imbalance: imbalanced-learn (SMOTE)
- Tuning: optuna
- Explainability: shap
- Tracking: mlflow
- API: fastapi, uvicorn
- Container: docker
- CI/CD: GitHub Actions
- Visualization: matplotlib, seaborn

## Project Structure

```
Loan_prediction/
├── src/
│   └── loan_prediction.py   # Full pipeline
├── loan_api/
│   ├── main.py                    # FastAPI app
│   ├── Dockerfile
│   └── requirements.txt
├── outputs/
│   ├── feature_importance.csv
│   └── plots/
├── tests/
│   └── test_features.py           # 8 unit tests
├── .github/
│   └── workflows/
│       └── test.yml               # CI/CD
├── requirements.txt
└── .gitignore
```

## How to Run

```bash
git clone https://github.com/Thinuridesilva/Loan_prediction.git
cd Loan_prediction
pip install -r requirements.txt
python src/loan_default_pipeline.py
```

Run tests:
```bash
pytest tests/ -v
```

Run API locally:
```bash
cd loan_api
pip install -r requirements.txt
uvicorn main:app --reload
```

Run API with Docker:
```bash
cd loan_api
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

Example prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "Income": 60000, "LoanAmount": 20000, "CreditScore": 650,
       "MonthsEmployed": 24, "NumCreditLines": 3, "InterestRate": 12.5,
       "LoanTerm": 36, "DTIRatio": 0.4, "Education": "Bachelor'\''s",
       "EmploymentType": "Full-time", "MaritalStatus": "Single",
       "HasMortgage": "No", "HasDependents": "No",
       "LoanPurpose": "Auto", "HasCoSigner": "No"}'
```

Example response:
```json
{
  "default_probability": 0.1685,
  "prediction": 1,
  "risk_level": "MEDIUM",
  "threshold_used": 0.06
}
```

View MLflow experiments:
```bash
mlflow ui
```
Open http://localhost:5000
