if request.method == "POST":
    # 1. Gather inputs as before
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

    # Encode binary features
    input_df['gender'] = input_df['gender'].map({'Male':1,'Female':0})
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        input_df[col] = input_df[col].map({'Yes':1,'No':0})

    # Encode categorical features using encoders
    for column, encoder in encoders.items():
        if column in input_df.columns and input_df[column].dtype=='object':
            input_df[column] = encoder.transform(input_df[column])

    # Reorder columns to match training
    training_cols = model.get_booster().feature_names  # or hardcode list
    input_df = input_df[training_cols]

    # Ensure numeric types
    input_df = input_df.astype(float)

    # Fill missing
    input_df.fillna(0, inplace=True)

    # Predict safely
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
    except Exception as e:
        print("Prediction Error:", e)
        return render_template("index.html", result=f"Error: {e}")

    result = "WILL CHURN" if prediction==1 else "WILL STAY"
    confidence = f"{max(proba)*100:.1f}%"
    risk = "High" if proba[1]>0.7 else "Medium" if proba[1]>0.4 else "Low"
    chart = create_prob_chart(proba)


