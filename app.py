from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained linear regression model
model_path = "model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get median income from the form
        median_income = float(request.form.get("median_income"))
        # Prepare data for prediction
        input_data = np.array([[median_income]])
        # Predict house price
        predicted_price = model.predict(input_data)[0]
        return jsonify({
            "status": "success",
            "predicted_price": round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)