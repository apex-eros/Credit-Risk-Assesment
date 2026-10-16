import joblib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from backend.shap_utils import generate_waterfall
from pathlib import Path
from config.mapping import sex_map,saving_map,checking_map,purpose_map
from schema.user_input import CustomerData
from models.prediction import prediction

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

app = FastAPI(title="Credit Risk Prediction API",description="Predict customer default risk",
              version="1.0")

app.mount(
    "/assets",
    StaticFiles(directory="assets"),
    name="assets"
)

@app.get('/')
def home():
    return {"message": "Credit Risk Prediction API Running"}

@app.post('/predict')
def predict(data: CustomerData):
    input_df = {
         "Age": data.Age,
        "Sex": data.Sex,
        "Job": data.Job,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Housing_own": data.Housing_own,
        "Housing_rent": data.Housing_rent,
        "Purpose" : data.Purpose
    }

    result = prediction(model, input_df)
    
    return {
        "customer_details": {
        "age": data.Age,
        "sex": sex_map[data.Sex],
        "saving_account": saving_map[data.Saving_accounts],
        "checking_account": checking_map[data.Checking_account],
        "purpose" : purpose_map[data.Purpose]
    },
    **result
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
        "Housing_rent": data.Housing_rent,
        "Purpose" : data.Purpose
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