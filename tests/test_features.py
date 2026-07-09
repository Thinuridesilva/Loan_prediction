import pandas as pd
import numpy as np
import pytest

def make_sample_df():
    return pd.DataFrame({
        'Age'            : [25, 45, 60],
        'Income'         : [50000, 80000, 120000],
        'LoanAmount'     : [10000, 40000, 80000],
        'CreditScore'    : [600, 720, 800],
        'MonthsEmployed' : [12, 60, 120],
        'NumCreditLines' : [2, 4, 6],
        'InterestRate'   : [5.5, 10.2, 15.0],
        'LoanTerm'       : [12, 36, 60],
        'DTIRatio'       : [0.2, 0.4, 0.6],
        'Education'      : ["Bachelor's", "Master's", 'PhD'],
        'EmploymentType' : ['Full-time', 'Part-time', 'Self-employed'],
        'MaritalStatus'  : ['Single', 'Married', 'Divorced'],
        'HasMortgage'    : ['Yes', 'No', 'Yes'],
        'HasDependents'  : ['No', 'Yes', 'No'],
        'LoanPurpose'    : ['Home', 'Auto', 'Education'],
        'HasCoSigner'    : ['No', 'Yes', 'No'],
        'Default'        : [0, 1, 0],
    })

def test_loan_to_income():
    df = make_sample_df()
    df['LoanToIncome'] = df['LoanAmount'] / (df['Income'] + 1)
    expected = 10000 / (50000 + 1)
    assert abs(df['LoanToIncome'].iloc[0] - expected) < 1e-6

def test_employed_years():
    df = make_sample_df()
    df['EmployedYears'] = df['MonthsEmployed'] / 12
    assert df['EmployedYears'].iloc[0] == 1.0
    assert df['EmployedYears'].iloc[1] == 5.0

def test_debt_load():
    df = make_sample_df()
    df['DebtLoad'] = df['DTIRatio'] * df['Income']
    assert df['DebtLoad'].iloc[0] == 0.2 * 50000

def test_education_encoding():
    df = make_sample_df()
    edu_map = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    df['Education'] = df['Education'].map(edu_map)
    assert df['Education'].iloc[0] == 1
    assert df['Education'].iloc[1] == 2
    assert df['Education'].iloc[2] == 3

def test_binary_encoding():
    df = make_sample_df()
    for col in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    assert df['HasMortgage'].iloc[0] == 1
    assert df['HasMortgage'].iloc[1] == 0

def test_no_missing_values():
    df = make_sample_df()
    df['LoanToIncome']       = df['LoanAmount'] / (df['Income'] + 1)
    df['LoanAmountPerMonth'] = df['LoanAmount'] / (df['LoanTerm'] + 1)
    df['IncomePerMonth']     = df['Income'] / 12
    df['DebtLoad']           = df['DTIRatio'] * df['Income']
    df['CreditUtilization']  = df['LoanAmount'] / (df['CreditScore'] + 1)
    df['EmployedYears']      = df['MonthsEmployed'] / 12
    assert df.isnull().sum().sum() == 0

def test_target_values():
    df = make_sample_df()
    assert set(df['Default'].unique()).issubset({0, 1})

def test_loan_to_income_positive():
    df = make_sample_df()
    df['LoanToIncome'] = df['LoanAmount'] / (df['Income'] + 1)
    assert (df['LoanToIncome'] > 0).all()
