import numpy as np
import sklearn
import flask
import joblib
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Flask imported successfully")
print("All imports successful!")

from sklearn.linear_model import LinearRegression
import joblib

# Training data
X = [[1], [2], [3], [4], [5], [6]]
y = [20, 30, 40, 50, 60, 70]
# Create and train model
model = LinearRegression()
model.fit(X, y)
# Save the model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved successfully!")

#Create the Flask API.
from flask import Flask, request, jsonify
import joblib

import threading
# Load the trained model
model = joblib.load("student_model.pkl")
# Create Flask application
app = Flask(__name__)
@app.route("/")
def home():
    return "ML Model Deployment using Flask is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    hours = float(data["hours"])
    prediction = model.predict([[hours]])
    return jsonify({
        "study_hours": hours,
        "predicted_marks": round(float(prediction[0]), 2)
})

# Start Flask server in the background
def run_app():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

thread = threading.Thread(target=run_app, daemon=True)
thread.start()
print("Flask server started!")

#Test the deployed model
import requests
url = "http://127.0.0.1:5000/predict"
data = {"hours": 7}
response = requests.post(url, json=data)
print(response.json())
