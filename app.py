from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('salary_model.pkl')
  # Load the magic formula

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    years_exp = data.get('years_experience')
    if years_exp is None:
        return jsonify({'error': 'Please provide years_experience'}), 400
    prediction = model.predict([[years_exp]])
    return jsonify({'predicted_salary': float(prediction[0])})

