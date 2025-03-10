import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer with pipelines for numerical and categorical features.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",  # Must exactly match the CSV header
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
                # Set with_mean=False because OneHotEncoder returns sparse matrices by default.
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Created numerical and categorical pipelines.")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test data, applies the preprocessing pipeline, and saves the preprocessor object.
        Returns the transformed train and test arrays along with the preprocessor file path.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            preprocessor = self.get_data_transformer_object()
            target_column = "math_score"

            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            logging.info("Features and target split completed.")

            # Fit on training data and transform both training and testing data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Data transformation (fit/transform) completed.")

            # Combine transformed features with target
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save the preprocessor object for future use
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info("Preprocessor object saved successfully.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
