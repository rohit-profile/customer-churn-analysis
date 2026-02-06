import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Load model and encoders
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
    return model, encoders

model, encoders = load_model()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

# Sidebar inputs
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tenure = st.sidebar.number_input("Tenure (months)", min_value=0)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0)

if st.button("🔍 Predict Churn"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Binary encoding
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # Categorical encoding
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    result = "🚨 WILL CHURN" if prediction == 1 else "✅ WILL STAY"
    confidence = f"{max(proba) * 100:.1f}%"
    risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"

    st.subheader("📌 Prediction Result")
    st.write(f"**Result:** {result}")
    st.write(f"**Confidence:** {confidence}")
    st.write(f"**Risk Level:** {risk}")

    # Plotly chart
    fig = go.Figure(data=[
        go.Bar(name='No Churn', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
        go.Bar(name='Churn', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        barmode='group',
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)
