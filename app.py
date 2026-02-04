from flask import Flask, render_template, request 
import pandas as pd 
import pickle 
import plotly 
import plotly.graph_objects as go 
import json


app = Flask(__name__)

# Load model and encoders
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_model()

# Helper function to create Plotly bar chart JSON
def create_prob_chart(proba):
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
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    risk = None
    chart = None

    if model is None or encoders is None:
        return "Model or encoders not found. Please generate model files first."

    if request.method == "POST":
        # Get form inputs
        gender = request.form["gender"]
        senior_citizen = int(request.form["senior_citizen"])
        partner = request.form["partner"]
        dependents = request.form["dependents"]
        phone_service = request.form["phone_service"]
        multiple_lines = request.form["multiple_lines"]
        internet_service = request.form["internet_service"]
        online_security = request.form["online_security"]
        online_backup = request.form["online_backup"]
        device_protection = request.form["device_protection"]
        tech_support = request.form["tech_support"]
        streaming_tv = request.form["streaming_tv"]
        streaming_movies = request.form["streaming_movies"]
        tenure = int(request.form["tenure"])
        contract = request.form["contract"]
        paperless_billing = request.form["paperless_billing"]
        payment_method = request.form["payment_method"]
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])

        # Prepare DataFrame
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

        # Encode binary features
        input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

        # Encode categorical features using saved encoders
        for column, encoder in encoders.items():
            if column in input_df.columns and input_df[column].dtype == 'object':
                input_df[column] = encoder.transform(input_df[column])

        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Prepare output
        result = "WILL CHURN" if prediction == 1 else "WILL STAY"
        confidence = f"{max(proba) * 100:.1f}%"
        risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
        chart = create_prob_chart(proba)

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           risk=risk,
                           chart=chart)

if __name__ == "__main__":
    app.run(debug=True)

