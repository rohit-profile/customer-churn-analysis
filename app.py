from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = risk = proba = None

    if request.method == "POST":
        try:
            input_data = {
                'gender': request.form.get('gender'),
                'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
                'Partner': request.form.get('Partner'),
                'Dependents': request.form.get('Dependents'),
                'tenure': int(request.form.get('tenure', 0)),
                'PhoneService': request.form.get('PhoneService'),
                'MultipleLines': request.form.get('MultipleLines'),
                'InternetService': request.form.get('InternetService'),
                'OnlineSecurity': request.form.get('OnlineSecurity'),
                'OnlineBackup': request.form.get('OnlineBackup'),
                'DeviceProtection': request.form.get('DeviceProtection'),
                'TechSupport': request.form.get('TechSupport'),
                'StreamingTV': request.form.get('StreamingTV'),
                'StreamingMovies': request.form.get('StreamingMovies'),
                'Contract': request.form.get('Contract'),
                'PaperlessBilling': request.form.get('PaperlessBilling'),
                'PaymentMethod': request.form.get('PaymentMethod'),
                'MonthlyCharges': float(request.form.get('MonthlyCharges', 0)),
                'TotalCharges': float(request.form.get('TotalCharges', 0) or 0)
            }

            df = pd.DataFrame([input_data])

        
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
            for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                df[col] = df[col].map({'Yes': 1, 'No': 0})

            # -------- Safe Encoder Transform --------
            for col, encoder in encoders.items():
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0]
                        if x in encoder.classes_
                        else -1
                    )

          
            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0]

            prediction = "WILL CHURN 🚨" if pred == 1 else "WILL STAY ✅"
            confidence = f"{round(max(proba)*100, 1)}%"
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"

        except Exception as e:
            prediction = "ERROR ❌"
            confidence = "-"
            risk = "Check Inputs"
            print("ERROR:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        proba=proba
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
