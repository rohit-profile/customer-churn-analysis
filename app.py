from flask import Flask, render_template, request
import pandas as pd
import pickle
import plotly
import plotly.graph_objects as go
import json

app = Flask(__name__)

# -------------------------------
# Load model and encoders safely
# -------------------------------
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_model()

# -------------------------------
# Plotly bar chart helper
# -------------------------------
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

# -------------------------------
# Main route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    risk = None
    chart = None

    if model is None or encoders is None:
        return "Model or encoders not found. Please generate model files first."

  if request.method == "POST":
    # -------------------------------
    # Step 1: Gather form input
    # -------------------------------
    input_data = {
        'gender': request.form["gender"],
        'SeniorCitizen': int(request.form["senior_citizen"]),
        'Partner': request.form["partner"],
        'Dependents': request.form["dependents"],
        'tenure': int(request.form["tenure"]),
        'PhoneService': request.form["phone_service"],
        'MultipleLines': request.form["multiple_lines"],
        'InternetService': request.form["internet_service"],
        'OnlineSecurity': request.form["online_security"],
        'OnlineBackup': request.form["online_backup"],
        'DeviceProtection': request.form["device_protection"],
        'TechSupport': request.form["tech_support"],
        'StreamingTV': request.form["streaming_tv"],
        'StreamingMovies': request.form["streaming_movies"],
        'Contract': request.form["contract"],
        'PaperlessBilling': request.form["paperless_billing"],
        'PaymentMethod': request.form["payment_method"],
        'MonthlyCharges': float(request.form["monthly_charges"]),
        'TotalCharges': float(request.form["total_charges"])
    }

    input_df = pd.DataFrame([input_data])

    # -------------------------------
    # Step 2: Encode binary features
    # -------------------------------
    input_df['gender'] = input_df['gender'].map({'Male':1,'Female':0})
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        input_df[col] = input_df[col].map({'Yes':1,'No':0})

    # -------------------------------
    # Step 3: Encode remaining categorical features
    # -------------------------------
    for column, encoder in encoders.items():
        if column in input_df.columns and input_df[column].dtype == 'object':
            input_df[column] = encoder.transform(input_df[column])

    # -------------------------------
    # Step 4: Ensure numeric columns only are float
    # -------------------------------
    numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','gender','Partner','Dependents','PhoneService','PaperlessBilling']
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)

    # -------------------------------
    # Step 5: Reorder columns to match training
    # -------------------------------
    training_cols = model.get_booster().feature_names
    input_df = input_df[training_cols]

    # -------------------------------
    # Step 6: Fill missing values
    # -------------------------------
    input_df.fillna(0, inplace=True)

    # -------------------------------
    # Step 7: Predict safely
    # -------------------------------
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", result=f"Error: {e}")

    # -------------------------------
    # Step 8: Prepare output
    # -------------------------------
    result = "WILL CHURN" if prediction == 1 else "WILL STAY"
    confidence = f"{max(proba)*100:.1f}%"
    risk = "High" if proba[1]>0.7 else "Medium" if proba[1]>0.4 else "Low"
    chart = create_prob_chart(proba)

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           risk=risk,
                           chart=chart)


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
