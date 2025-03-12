import sys
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def train_pipeline():
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion complete.")

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation complete.")

        trainer = ModelTrainer()
        best_model_name, best_score, results, predictions = trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
        logging.info(f"Training complete. Best model: {best_model_name} with R²: {best_score:.4f}")
        print(f"Best model: {best_model_name}, R²: {best_score:.4f}")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_pipeline()