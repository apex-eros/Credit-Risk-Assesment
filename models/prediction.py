
import pandas as pd


def prediction(model, input_data: dict) -> dict:
    """Predict customer class and calculate risk details."""

    # Convert dictionary to a DataFrame with one row
    input_df = pd.DataFrame([input_data])

    predicted_class = int(model.predict(input_df)[0])

    # Find the probability corresponding to class 1
    class_index = list(model.classes_).index(1)
    probability = float(
        model.predict_proba(input_df)[0][class_index]
    )

    prediction_label = (
        "Good Customer" if predicted_class == 1
        else "Bad Customer"
    )

    if probability >= 0.7:
        risk_label = "Low Risk"
    elif probability >= 0.4:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    credit_score = round(300 + probability * 600)

    if credit_score >= 750:
        grade = "A"
    elif credit_score >= 650:
        grade = "B"
    elif credit_score >= 550:
        grade = "C"
    else:
        grade = "D"

    return {
        "prediction": prediction_label,
        "probability_good_customer": round(probability, 4),
        "risk_category": risk_label,
        "credit_score": credit_score,
        "risk_grade": grade,
    }
