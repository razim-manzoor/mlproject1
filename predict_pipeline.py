import sys
import pandas as pd
from src.logger import logging
from src.utils import load_object

def predict(input_data_path):
    try:
        preprocessor = load_object("artifacts/preprocessor.pkl")
        model = load_object("artifacts/model.pkl")
        input_df = pd.read_csv(input_data_path)
        transformed_data = preprocessor.transform(input_df)
        predictions = model.predict(transformed_data)
        logging.info(f"Predictions: {predictions[:10]}")
        print("Predictions:", predictions.tolist())
    except Exception as e:
        logging.error(f"Prediction pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_pipeline.py <input_data_csv>")
        sys.exit(1)
    predict(sys.argv[1])