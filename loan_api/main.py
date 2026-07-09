import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Loan Default Prediction API")

# Load model
model = joblib.load("xgb_tuned_model.pkl")

# Input schema
class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Feature engineering
    df["LoanToIncome"]       = df["LoanAmount"] / (df["Income"] + 1)
    df["LoanAmountPerMonth"] = df["LoanAmount"] / (df["LoanTerm"] + 1)
    df["IncomePerMonth"]     = df["Income"] / 12
    df["DebtLoad"]           = df["DTIRatio"] * df["Income"]
    df["CreditUtilization"]  = df["LoanAmount"] / (df["CreditScore"] + 1)
    df["EmployedYears"]      = df["MonthsEmployed"] / 12

    # Binary encoding
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Education ordinal
    edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df["Education"] = df["Education"].map(edu_map)

    # One-hot encode
    df = pd.get_dummies(df, columns=["EmploymentType", "MaritalStatus", "LoanPurpose"], drop_first=True)

    # Align columns with training data
    trained_cols = model.get_booster().feature_names
    for col in trained_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_cols]

    return df

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API", "status": "running"}

@app.post("/predict")
def predict(application: LoanApplication):
    data = application.dict()
    df = preprocess(data)
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= 0.06)  # optimal threshold
    return {
        "default_probability": round(float(proba), 4),
        "prediction": prediction,
        "risk_level": "HIGH" if proba >= 0.3 else "MEDIUM" if proba >= 0.1 else "LOW",
        "threshold_used": 0.06
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
