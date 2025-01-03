from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")  # Replace with your actual model file path

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    median_income = data.get('median_income')

    if median_income is None:
        return jsonify({'error': 'Median income is required'}), 400

    try:
        # Convert input to float and predict
        median_income = float(median_income)
        prediction = model.predict(np.array([[median_income]]))  # Adjust based on your model input shape
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
