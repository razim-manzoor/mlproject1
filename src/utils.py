import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Train and evaluate each model provided in the models dictionary using R² score.
    
    Parameters:
        X_train, y_train: Training features and target.
        X_test, y_test: Testing features and target.
        models (dict): A dictionary mapping model names to model instances.
    
    Returns:
        results (dict): Mapping of model names to their R² scores.
        best_model_name (str): The name of the best performing model.
        best_model: The best performing model instance.
        best_score (float): The best R² score.
    """
    best_model_name = None
    best_model = None
    best_score = -float("inf")
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        results[name] = score
        logging.info(f"{name} R² score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    return results, best_model_name, best_model, best_score