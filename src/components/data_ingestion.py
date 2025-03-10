import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw CSV file, saves a copy as raw data, splits it into train and test sets,
        and saves them to disk.
        """
        logging.info("Starting data ingestion process.")
        try:
            # Adjust this path to where your CSV file is actually located.
            data_path = os.path.join('src', 'notebook', 'data', 'stud.csv')
            df = pd.read_csv(data_path)
            logging.info("Dataset loaded successfully.")

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully.")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train-test split completed and files saved.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion process finished successfully.")

        # Optionally, call the data transformation process next
        from src.components.data_transformation import DataTransformation
        transformation = DataTransformation()
        transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation process finished successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)
