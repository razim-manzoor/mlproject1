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
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Train multiple regression models, evaluate them with R² score, save the best model,
        and return evaluation results and predictions on X_test.
        """
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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

            results, best_model_name, best_model, best_score = evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Best model: {best_model_name} with R² score: {best_score:.4f}")

            # Optional: enforce a minimum performance threshold.
            if best_score < 0.6:
                raise CustomException("No model achieved acceptable performance.", sys)

            save_object(self.config.trained_model_path, best_model)
            logging.info("Best model saved successfully.")

            predictions = best_model.predict(X_test)
            logging.info(f"Predictions on X_test: {predictions}")

            return best_model_name, best_score, results, predictions

        except Exception as e:
            raise CustomException(e, sys)