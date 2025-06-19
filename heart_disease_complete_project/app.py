
from flask import Flask, render_template, request, flash
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'
model = joblib.load('random_forest_heart_model.pkl')

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    confidence = None
    form_data = {}

    if request.method == 'POST':
        try:
            for field in feature_names:
                form_data[field] = request.form.get(field, '')
            input_values = [float(form_data[field]) for field in feature_names]
            input_df = pd.DataFrame([input_values], columns=feature_names)

            prob = model.predict_proba(input_df)[0][1]
            prediction = 1 if prob >= 0.55 else 0
            confidence = round(prob * 100, 2)

            prediction_text = (
                f"<span class='text-danger fw-bold'>❤️ The person has heart disease</span>"
                if prediction == 1
                else f"<span class='text-success fw-bold'>🟢 The person does not have heart disease</span>"
            )
        except Exception as e:
            flash(f"Error: {str(e)}")

    return render_template("index.html", prediction_text=prediction_text,
                           confidence=confidence, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
