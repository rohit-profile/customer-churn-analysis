from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
ENCODER_PATH = os.path.join(os.getcwd(), "encoders.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENCODER_PATH, "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    risk = None
    proba = None

    if request.method == "POST":
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        df = pd.DataFrame([input_data])

        # Binary encoding
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        # Encoder transform
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Prediction
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        prediction = "WILL CHURN 🚨" if pred == 1 else "WILL STAY ✅"
        confidence = f"{max(proba) * 100:.1f}%"
        risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        proba=proba
    )

if __name__ == "__main__":
    app.run(debug=True)
