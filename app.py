from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load('salary_model.pkl')  # Load your trained model

@app.route('/')
def home():
    return "Salary Prediction API is running. Use /predict with POST method."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    years_exp = data.get('years_experience')
    if years_exp is None:
        return jsonify({'error': 'Please provide years_experience'}), 400
    prediction = model.predict([[years_exp]])
    return jsonify({'predicted_salary': float(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



