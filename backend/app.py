import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from backend.shap_utils import generate_waterfall
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

app = FastAPI(title="Credit Risk Prediction API",description="Predict customer default risk",
              version="1.0")

app.mount(
    "/assets",
    StaticFiles(directory="assets"),
    name="assets"
)

sex_map = {
    0: "Female",
    1: "Male"
}

saving_map = {
    0: "Little",
    1: "Moderate",
    2: "Quite Rich",
    3: "Rich"
}

checking_map = {
    0: "Little",
    1: "Moderate",
    2: "Rich"
}

class CustomerData(BaseModel):
    Age: int
    Sex: int
    Job: int
    Saving_accounts: int
    Checking_account: int
    Credit_amount: float
    Duration: int
    Housing_own: int
    Housing_rent: int

@app.get('/')
def home():
    return {"message": "Credit Risk Prediction API Running"}

@app.post('/predict')
def predict(data: CustomerData):
    input_df = pd.DataFrame([{
         "Age": data.Age,
        "Sex": data.Sex,
        "Job": data.Job,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Housing_own": data.Housing_own,
        "Housing_rent": data.Housing_rent
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    prediction_label = (
    "Good Customer"
    if prediction == 1
    else "Bad Customer"
)

    if probability >= 0.7:
        risk_label = "Low Risk"
    elif probability >= 0.4:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    credit_score = int(300 + probability * 600)

    if credit_score >= 750:
        grade = "A"
    elif credit_score >= 650:
        grade = "B"
    elif credit_score >= 550:
        grade = "C"
    else:
        grade = "D"

    return {
    "customer_details": {
        "age": data.Age,
        "sex": sex_map[data.Sex],
        "saving_account": saving_map[data.Saving_accounts],
        "checking_account": checking_map[data.Checking_account]
    },
    "prediction": prediction_label,
    "probability_good_customer": round(float(probability), 4),
    "risk_category": risk_label,
    "credit_score": credit_score,
    "risk_grade": grade
    
}


@app.post("/explain")
def explain(data: CustomerData):

    customer_df = pd.DataFrame([{
        "Age": data.Age,
        "Sex": data.Sex,
        "Job": data.Job,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Housing_own": data.Housing_own,
        "Housing_rent": data.Housing_rent
    }])

    generate_waterfall(customer_df)

    return {
        "message": "SHAP generated"
    }

@app.get("/metrics")
def metrics():

    return {
        "roc_auc": 0.72,
        "f1": 0.73,
        "ks": 0.38,
        "gini": 0.44
    }