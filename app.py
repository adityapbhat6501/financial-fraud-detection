from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained Random Forest model
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert Time from HH:MM to seconds
        time_str = request.form['time']
        h, m = map(int, time_str.split(":"))
        time_in_seconds = h*3600 + m*60

        # Amount input
        amount = float(request.form['amount'])

        # User-friendly categorical inputs
        transaction_type = request.form['trans_type']  # Online, POS, ATM
        country_risk = request.form['country_risk']   # Low, Medium, High
        device = request.form['device']               # Mobile, Desktop, Other
        previous_history = request.form['previous_history']  # Yes/No

        # Map Transaction Type to V1–V5
        if transaction_type == "Online":
            v1, v2, v3, v4, v5 = 2.5, 0.5, -1.0, 0.0, 0.3
        elif transaction_type == "POS":
            v1, v2, v3, v4, v5 = -0.5, 1.0, 0.2, -0.3, 0.1
        else:  # ATM
            v1, v2, v3, v4, v5 = 0.0, -0.5, 1.2, 0.3, -0.2

        # Map Country Risk to V6–V10
        if country_risk == "Low":
            v6, v7, v8, v9, v10 = 0.1, 0.2, 0.0, 0.1, 0.0
        elif country_risk == "Medium":
            v6, v7, v8, v9, v10 = 0.5, 0.4, -0.2, 0.3, 0.1
        else:  # High
            v6, v7, v8, v9, v10 = 1.0, 0.8, 0.5, 0.6, 0.4

        # Map Device to V11–V15
        if device == "Mobile":
            v11, v12, v13, v14, v15 = 0.2, 0.1, 0.0, 0.3, 0.0
        elif device == "Desktop":
            v11, v12, v13, v14, v15 = -0.1, 0.0, 0.2, -0.2, 0.1
        else:  # Other
            v11, v12, v13, v14, v15 = 0.0, 0.1, 0.3, 0.0, 0.2

        # Map Previous History to V16–V20
        if previous_history == "No":
            v16, v17, v18, v19, v20 = 0.0, 0.0, 0.0, 0.0, 0.0
        else:  # Yes
            v16, v17, v18, v19, v20 = 1.0, 1.0, 0.8, 0.9, 0.7

        # Remaining features V21–V28: small random values to simulate variability
        v_remaining = np.random.uniform(-3, 3, 8).tolist()

        # Construct final input vector
        input_data = np.array(
            [time_in_seconds, v1, v2, v3, v4, v5,
             v6, v7, v8, v9, v10,
             v11, v12, v13, v14, v15,
             v16, v17, v18, v19, v20] + v_remaining + [amount]
        ).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        result = "⚠️ Fraud Detected" if prediction == 1 else "✅ Transaction is Safe"

    except Exception as e:
        result = f"Error: {e}"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
