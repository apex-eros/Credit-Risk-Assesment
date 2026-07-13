# Credit Risk Assessment System

A complete end-to-end **Machine Learning Credit Risk Assessment Platform** built using **XGBoost, FastAPI, Streamlit, SHAP Explainability, and AWS EC2 Deployment**.

The application predicts whether a customer is likely to be a **Good Customer** or **Bad Customer**, provides a custom **Credit Score**, **Risk Grade**, and explains the prediction using **SHAP Explainability**.

---

## 📌 Project Overview

Financial institutions need to assess the risk associated with lending money to customers.

This project uses Machine Learning to evaluate customer credit risk based on demographic and financial attributes.

The system provides:

* Credit Risk Prediction
* Probability of Good Customer
* Risk Categorization
* Credit Score Estimation
* Credit Grade Assignment
* SHAP Explainability
* Interactive Web Interface
* REST API Deployment

---

## System Architecture

```text
┌────────────────────┐
│     Streamlit UI   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│    FastAPI API     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   XGBoost Model    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ SHAP Explainability│
└────────────────────┘
```

---

# Project Structure

```text
Credit-Risk-Assessment/

│
├── backend/
│   ├── app.py
│   ├── shap_utils.py
│
├── frontend/
│   └── streamlit_app.py
│
├── models/
│   └── xgb_model.pkl
│
├── assets/
│   └── waterfall.png
│
├── data/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# Dataset Features

The model uses the following customer attributes:

| Feature          | Description             |
| ---------------- | ----------------------- |
| Age              | Customer Age            |
| Sex              | Male/Female             |
| Job              | Job Category            |
| Saving Accounts  | Savings Account Status  |
| Checking Account | Checking Account Status |
| Credit Amount    | Requested Credit Amount |
| Duration         | Loan Duration           |
| Housing Own      | Own House               |
| Housing Rent     | Rental House            |

---

# Target Variable

```python
Risk
```

| Value | Meaning       |
| ----- | ------------- |
| 1     | Good Customer |
| 0     | Bad Customer  |

---

# Machine Learning Pipeline

The project follows a complete ML lifecycle:

### 1. Data Cleaning

* Missing Value Handling
* Encoding Categorical Variables
* Feature Engineering

### 2. Model Training

Models Evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* XGBoost

### 3. Hyperparameter Tuning

Used:

```python
GridSearchCV
```

with:

```python
scoring="roc_auc"
```

---

## Experiment Tracking

MLflow was used for:

* Hyperparameter Tracking
* Metrics Logging
* Model Versioning
* Artifact Storage

Logged Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

---

# Final Model

Model Selected:

```python
XGBoost Classifier
```

Reason:

* Highest ROC-AUC
* Strong Generalization
* Better Recall
* Better Separation Between Good and Bad Customers

---

# Model Evaluation Metrics

## ROC-AUC

Measures model's ability to distinguish:

```text
Good Customers
vs
Bad Customers
```

Range:

```text
0.5 = Random Guess
1.0 = Perfect Model
```

---

## KS Statistic

Measures separation between:

```text
Good Customers
vs
Bad Customers
```

Formula:

```text
KS = Max(TPR - FPR)
```

Interpretation:

| KS Score | Quality   |
| -------- | --------- |
| <20      | Weak      |
| 20-40    | Good      |
| 40-60    | Very Good |
| >60      | Excellent |

---

## Gini Coefficient

Derived from ROC-AUC.

Formula:

```text
Gini = 2 × AUC − 1
```

Interpretation:

| Gini    | Performance |
| ------- | ----------- |
| <0.2    | Poor        |
| 0.2-0.4 | Fair        |
| 0.4-0.6 | Good        |
| >0.6    | Excellent   |

---

# 🔍 SHAP Explainability

The project uses SHAP to explain model predictions.

Implemented:

### Global Explainability

Shows:

* Most Important Features
* Overall Feature Impact

Example:

```python
shap.summary_plot()
```

---

### Local Explainability

Explains individual customer decisions.

Example:

```python
shap.waterfall_plot()
```

Provides:

* Positive Contributors
* Negative Contributors
* Final Prediction Reasoning

---

# FastAPI Backend

The backend exposes REST APIs.

---

## Health Check

### GET /

```json
{
  "message": "Credit Risk Prediction API Running"
}
```

---

## Prediction Endpoint

### POST /predict

Request:

```json
{
  "Age":22,
  "Sex":0,
  "Job":2,
  "Saving_accounts":1,
  "Checking_account":2,
  "Credit_amount":15000,
  "Duration":40,
  "Housing_own":0,
  "Housing_rent":1
}
```

Response:

```json
{
  "prediction":"Bad Customer",
  "probability_good_customer":0.2749,
  "risk_category":"High Risk",
  "credit_score":464,
  "risk_grade":"D"
}
```

---

## Explainability Endpoint

### POST /explain

Generates:

```text
SHAP Waterfall Plot
```

for the selected customer.

---

# 🎨 Streamlit Frontend

The frontend provides an interactive interface.

### Prediction Tab

Allows users to:

* Enter customer information
* Predict customer risk
* View credit score
* View risk grade

---

### SHAP Explainability Tab

Displays:

* Individual Waterfall Plot
* Feature Contributions

---

### Metrics Tab

Displays:

* ROC-AUC
* KS Statistic
* Gini Coefficient
* F1 Score

---

### What-If Analysis Tab

Allows simulation of:

* Different Loan Amounts
* Different Durations
* Different Customer Profiles

to observe prediction changes.

---

# 💳 Custom Credit Score

A simple credit score is generated from model probability.

Formula:

```python
credit_score = int(300 + probability * 600)
```

Range:

```text
300 → 900
```

---

## Credit Grades

| Score     | Grade |
| --------- | ----- |
| 750+      | A     |
| 650-749   | B     |
| 550-649   | C     |
| Below 550 | D     |

---

# ☁️ AWS Deployment

Deployed on:

### Amazon EC2

Services Running:

#### FastAPI

```bash
uvicorn backend.app:app \
--host 0.0.0.0 \
--port 8000
```

#### Streamlit

```bash
streamlit run frontend/streamlit_app.py \
--server.port 8501 \
--server.address 0.0.0.0
```

---

# 🛠️ Installation

Clone repository:

```bash
git clone <repo-url>
cd Credit-Risk-Assessment
```

Create environment:

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Backend

```bash
uvicorn backend.app:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

---

# ▶️ Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

# 🔮 Future Improvements

* Docker Deployment
* CI/CD Pipeline using GitHub Actions
* PostgreSQL Integration
* Model Monitoring
* Drift Detection
* Real Credit Bureau Score Integration
* PDF Credit Report Generation
* LLM-Based Decision Explanation (Groq/OpenAI)
* User Authentication
* Loan Approval Recommendation Engine

---

# 📚 Tech Stack

### Machine Learning

* Scikit-Learn
* XGBoost
* SHAP

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Experiment Tracking

* MLflow

### Deployment

* AWS EC2

---

# 👨‍💻 Author

**Jayesh Gangi**

Data Scientist | Machine Learning Engineer | Product Analytics

Interested in:

* Credit Risk Modeling
* Machine Learning Systems
* MLOps
* Generative AI
* Financial Analytics

