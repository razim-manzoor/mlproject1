from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
from src.utils import load_object
from src.logger import logging

app = Flask(__name__)

# Load preprocessor and model from artifacts
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH = os.path.join("artifacts", "model.pkl")
preprocessor = load_object(PREPROCESSOR_PATH)
model = load_object(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        # Convert numeric inputs
        data["writing_score"] = float(data["writing_score"])
        data["reading_score"] = float(data["reading_score"])
        # Convert form input to DataFrame
        input_df = pd.DataFrame([data])
        transformed_features = preprocessor.transform(input_df)
        predictions = model.predict(transformed_features)
        return render_template("index.html", prediction_text=f"Predicted math_score: {predictions[0]:.2f}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)