import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainning import ModelTrainerConfig
from src.components.model_trainning import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/Salary_Data.csv')
            logging.info('Read the dataset as dataframe')

            # Check for empty dataframe
            if df.empty:
                raise CustomException("Dataset is empty", sys)

            # Check for required columns
            required_columns = ['Age', 'Years of Experience', 'Gender', 'Education Level', 'Job Title', 'Salary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise CustomException(f"Missing required columns: {missing_columns}", sys)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Debug: Print split info
            logging.info(f'Train set shape: {train_set.shape}')
            logging.info(f'Test set shape: {test_set.shape}')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion complete')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f'Error in data ingestion: {str(e)}')
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    logging.info(f"Data ingestion and transformation completed successfully.")

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    logging.info(f"Model training completed successfully.")