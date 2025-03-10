import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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
    Train and evaluate models using R² score.
    
    Parameters:
        X_train, y_train: Training features and target.
        X_test, y_test: Testing features and target.
        models (dict): Dictionary mapping model names to model instances.
    
    Returns:
        results (dict): Mapping of model names to their R² scores.
        best_model_name (str): Name of the best performing model.
        best_model: The best performing model instance.
        best_score (float): Best R² score.
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

def tune_model(model, param_grid, X_train, y_train):
    """
    Tune the given model using GridSearchCV.
    
    Parameters:
        model: The machine learning model to tune.
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try.
        X_train, y_train: Training features and target.
    
    Returns:
        best_model: The best estimator from grid search.
        best_params (dict): The best parameter settings found.
        best_score (float): The best cross-validated R² score.
    """
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        raise CustomException(e, sys)