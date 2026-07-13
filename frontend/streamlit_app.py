import streamlit as st
import requests

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Risk Assessment System")
st.markdown("Predict customer credit risk using XGBoost")

# =====================================
# MAPPINGS
# =====================================

sex_map = {
    "Female": 0,
    "Male": 1
}

saving_map = {
    "Little": 0,
    "Moderate": 1,
    "Quite Rich": 2,
    "Rich": 3
}

checking_map = {
    "Little": 0,
    "Moderate": 1,
    "Rich": 2
}

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Prediction",
        "SHAP Explainability",
        "Metrics",
        "What-If Analysis"
    ]
)

# =====================================
# TAB 1 - PREDICTION
# =====================================

with tab1:

    st.header("Customer Information")

    with st.form("credit_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

            job = st.selectbox(
                "Job",
                [0, 1, 2, 3]
            )

            saving = st.selectbox(
                "Saving Account",
                ["Little", "Moderate", "Quite Rich", "Rich"]
            )

        with col2:

            checking = st.selectbox(
                "Checking Account",
                ["Little", "Moderate", "Rich"]
            )

            credit_amount = st.number_input(
                "Credit Amount",
                min_value=0.0,
                value=5000.0
            )

            duration = st.number_input(
                "Duration (Months)",
                min_value=1,
                value=12
            )

            housing = st.selectbox(
                "Housing",
                ["Own", "Rent","Free"]
            )

        predict_btn = st.form_submit_button("Predict Risk")

    payload = {
        "Age": int(age),
        "Sex": sex_map[sex],
        "Job": int(job),
        "Saving_accounts": saving_map[saving],
        "Checking_account": checking_map[checking],
        "Credit_amount": float(credit_amount),
        "Duration": int(duration),
        "Housing_own": 1 if housing == "Own" else 0,
        "Housing_rent": 1 if housing == "Rent" else 0
    }

    if predict_btn:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            result = response.json()

            st.success("Prediction Generated")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Probability Good Customer",
                    f"{result['probability_good_customer']:.2%}"
                )

            with col2:
                st.metric(
                    "Prediction",
                    result["prediction"]
                )

            with col3:
                st.metric(
                    "Risk Category",
                    result["risk_category"]
                )
            with col4:
                st.metric(
                    "Credit Score",
                    result["credit_score"]
                )

            with col5:
                st.metric(
                    "Grade",
                    result["risk_grade"]
                )

            st.subheader("Customer summary")

            st.json(result["customer_details"])

            st.subheader("Credit Score")

            st.progress(result["credit_score"] / 900)

            if result["risk_grade"] == "A":
                st.success("✅ Recommendation: APPROVE")

            elif result["risk_grade"] == "B":
                st.info("🟦 Recommendation: APPROVE WITH REVIEW")

            elif result["risk_grade"] == "C":
                st.warning("🟨 Recommendation: MANUAL REVIEW")

            else:
                st.error("🟥 Recommendation: REJECT")

        except Exception as e:

            st.error(f"Error connecting to API: {e}")

# =====================================
# TAB 2 - SHAP
# =====================================

with tab2:

    st.header("Model Explainability")

    st.write(
        "Generate SHAP explanation for the customer."
    )

    if st.button("Generate SHAP Explanation"):

        try:

            response = requests.post(
                "http://127.0.0.1:8000/explain",
                json=payload
            )

            st.success("SHAP Generated")
            st.image("http://127.0.0.1:8000/assets/waterfall.png")

        except Exception as e:

            st.error(f"Error: {e}")

# =====================================
# TAB 3 - METRICS
# =====================================

with tab3:

    st.header("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ROC AUC",
            "72%"
        )

    with col2:
        st.metric(
            "F1 Score",
            "73%"
        )

    with col3:
        st.metric(
            "KS Statistic",
            "38%"
        )

    with col4:
        st.metric(
            "Gini",
            "44%"
        )

    st.divider()

    st.subheader("Model Artifacts")

    try:

        st.image(
            "confusion_matrix.png",
            caption="Confusion Matrix"
        )

    except:
        st.warning(
            "confusion_matrix.png not found"
        )

# =====================================
# TAB 4 - WHAT IF ANALYSIS
# =====================================

with tab4:

    st.header("What-If Analysis")

    st.write(
        "Change variables and see impact on prediction."
    )

    new_duration = st.slider(
        "Duration",
        min_value=1,
        max_value=72,
        value=int(duration)
    )

    new_credit = st.slider(
        "Credit Amount",
        min_value=100,
        max_value=50000,
        value=int(credit_amount)
    )

    if st.button("Run What-If Analysis"):

        modified_payload = payload.copy()

        modified_payload["Duration"] = new_duration
        modified_payload["Credit_amount"] = new_credit

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=modified_payload
            )

            result = response.json()

            st.success("Updated Prediction")

            st.metric(
                "New Probability",
                f"{result['probability_good_customer']:.2%}"
            )

            st.metric(
                "New Risk",
                result["risk_category"]
            )

        except Exception as e:

            st.error(f"Error: {e}")