import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")

explainer = shap.TreeExplainer(model)


def generate_waterfall(customer_df):

    shap_values = explainer.shap_values(customer_df)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=customer_df.iloc[0],
            feature_names=customer_df.columns
        ),
        show=False
    )

    plt.savefig("assets/waterfall.png", bbox_inches="tight")
    plt.close()

    return shap_values