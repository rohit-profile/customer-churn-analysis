# Customer Churn Analysis

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Issues](https://img.shields.io/github/issues/[(https://github.com/rohit-profile/customer-churn-analysis))

## Project Overview
Customer churn is a critical metric for businesses, indicating the percentage of customers who stop using a company's products or services over a given period. This project analyzes customer data to identify patterns and factors influencing churn. Insights derived can help design strategies to retain customers and improve business growth.

---

## Live Deployment
You can explore the interactive dashboard for customer churn analysis here:  
[🔗 View Deployment]((https://customer-churn-analysis-sbqx.onrender.com))  

---

## Dataset
The dataset contains information about customer demographics, services, and account details.  

**Download the dataset:** [Customer Churn CSV](https://<your-dataset-link>.csv)  

| Column Name         | Description |
|--------------------|-------------|
| `gender`           | Gender of the customer (Male/Female) |
| `SeniorCitizen`    | Indicates if the customer is a senior (1) or not (0) |
| `Partner`          | Whether the customer has a partner (Yes/No) |
| `Dependents`       | Whether the customer has dependents (Yes/No) |
| `tenure`           | Number of months the customer has stayed with the company |
| `PhoneService`     | Whether the customer has phone service (Yes/No) |
| `MultipleLines`    | Whether the customer has multiple phone lines (Yes/No/No phone service) |
| `InternetService`  | Type of internet service (DSL/Fiber optic/No) |
| `OnlineSecurity`   | Whether the customer has online security service (Yes/No/No internet service) |
| `OnlineBackup`     | Whether the customer has online backup service (Yes/No/No internet service) |
| `DeviceProtection` | Whether the customer has device protection (Yes/No/No internet service) |
| `TechSupport`      | Whether the customer has tech support (Yes/No/No internet service) |
| `StreamingTV`      | Whether the customer has streaming TV service (Yes/No/No internet service) |
| `StreamingMovies`  | Whether the customer has streaming movies service (Yes/No/No internet service) |
| `Contract`         | Type of contract (Month-to-month/One year/Two year) |
| `PaperlessBilling` | Whether the customer uses paperless billing (Yes/No) |
| `PaymentMethod`    | Payment method (Electronic check/Mailed check/Bank transfer/Credit card) |
| `MonthlyCharges`   | Monthly charges for the customer |
| `TotalCharges`     | Total charges incurred by the customer |
| `Churn`            | Whether the customer churned (Yes/No) |

---

## Project Objectives
1. **Data Exploration** – Understand distributions, correlations, and patterns.  
2. **Feature Engineering** – Handle missing values and convert categorical variables.  
3. **Predictive Modeling** – Build machine learning models to predict churn.  
4. **Insights & Recommendations** – Identify key drivers of churn and actionable strategies.

---

## Methodology
1. **Data Cleaning** – Handle missing or inconsistent data.  
2. **Exploratory Data Analysis (EDA)** – Visualizations and trends analysis.  
3. **Feature Encoding** – Convert categorical variables using one-hot or label encoding.  
4. **Model Building** – Train models like Logistic Regression, Random Forest, XGBoost.  
5. **Evaluation** – Metrics include Accuracy, Precision, Recall, F1-score, ROC-AUC.  
6. **Interpretation** – Analyze feature importance to understand churn drivers.  

---

## Tools & Libraries
- **Python**  
- **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn**  
- **Scikit-learn**, **XGBoost**  
- **Jupyter Notebook / Streamlit**  

---
## Live Link--> (https://customer-churn-analysis-sbqx.onrender.com)

---
## Project Workflow
```mermaid
flowchart LR
    A[Data Collection] --> B[Data Cleaning]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering]
    D --> E[Model Building]
    E --> F[Model Evaluation]
    F --> G[Insights & Recommendations]


