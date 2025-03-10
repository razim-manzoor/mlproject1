import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, tune_model

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Trains, tunes, and evaluates regression models.
        Returns the best model name, its R² score, evaluation results, and test predictions.
        """
        try:
            # Split training and test arrays into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models to evaluate
            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(objective="reg:squarederror", random_state=42),
                "CatBoostRegressor": CatBoostRegressor(verbose=False, random_state=42)
            }

            # Evaluate models and pick the best based on R² score
            results, best_model_name, best_model, best_score = evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Initial best model: {best_model_name} with R² score: {best_score:.4f}")

            # Hyperparameter tuning for select models
            tuning_params = {
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "XGBRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "CatBoostRegressor": {
                    "iterations": [100, 200],
                    "depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2]
                }
            }

            # If best model is one we want to tune, perform hyperparameter tuning.
            if best_model_name in tuning_params:
                logging.info(f"Starting hyperparameter tuning for {best_model_name}.")
                best_model_tuned, best_params, tuned_score = tune_model(best_model, tuning_params[best_model_name], X_train, y_train)
                logging.info(f"Hyperparameter tuning completed. Best params: {best_params}, Best CV score: {tuned_score:.4f}")
                tuned_predictions = best_model_tuned.predict(X_test)
                from sklearn.metrics import r2_score
                tuned_r2 = r2_score(y_test, tuned_predictions)
                logging.info(f"Tuned model R² on test set: {tuned_r2:.4f}")
                if tuned_r2 > best_score:
                    best_score = tuned_r2
                    best_model = best_model_tuned
                    best_model_name += "_tuned"
                    logging.info(f"Using tuned model: {best_model_name} with improved R²: {best_score:.4f}")

            if best_score < 0.6:
                raise CustomException("No model achieved acceptable performance.", sys)

            # Save the best model
            save_object(self.config.trained_model_path, best_model)
            logging.info("Best model saved successfully.")

            # Get predictions on test set
            predictions = best_model.predict(X_test)
            logging.info(f"First 10 predictions on X_test: {predictions[:10]}")

            return best_model_name, best_score, results, predictions

        except Exception as e:
            raise CustomException(e, sys)