from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = model.feature_names

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(probability, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
