import sys
from src.logger import logging

def main():
    try:
        # Data Ingestion
        from src.components.data_ingestion import DataIngestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")

        # Data Transformation
        from src.components.data_transformation import DataTransformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation completed successfully.")

        # Model Training with Hyperparameter Tuning
        from src.components.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        best_model_name, best_score, results, predictions = trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
        logging.info(f"Model training complete. Best model: {best_model_name} with R²: {best_score:.4f}")
        logging.info(f"First 10 predictions on test set: {predictions[:10]}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()