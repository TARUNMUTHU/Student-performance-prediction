from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import webbrowser
import threading

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model/model.pkl')
encoders = joblib.load('model/label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Collect input values from the form (excluding school)
            input_data = {
                'sex': request.form['sex'],
                'age': int(request.form['age']),
                'address': request.form['address'],
                'famsize': request.form['famsize'],
                'Pstatus': request.form['Pstatus'],
                'Medu': int(request.form['Medu']),
                'Fedu': int(request.form['Fedu']),
                'studytime': int(request.form['studytime']),
                'failures': int(request.form['failures']),
                'absences': int(request.form['absences']),
                'G1': int(request.form['G1']),
                'G2': int(request.form['G2']),
            }

            # Encode categorical inputs using saved encoders
            for field in ['sex', 'address', 'famsize', 'Pstatus']:
                le = encoders[field]
                input_data[field] = le.transform([input_data[field]])[0]

            # Define feature order (school removed)
            features = ['sex', 'age', 'address', 'famsize', 'Pstatus',
                        'Medu', 'Fedu', 'studytime', 'failures', 'absences', 'G1', 'G2']

            # Create DataFrame and predict
            X = pd.DataFrame([input_data])[features]
            result = model.predict(X)[0]
            prediction = "Pass ✅" if result == 1 else "Fail ❌"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

# Auto-open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False)
