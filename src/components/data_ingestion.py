import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def load_data(self):
        try:
            logging.info("Loading data from raw data path...")
            df = pd.read_csv("D:\Software\Portfolio\Student_Performance_Prediction\data\data.csv")

            logging.info('Reading data from raw data path is successful!')
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated!')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is successful!")
            return (self.config.train_data_path, self.config.test_data_path)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()  # Correct way to capture traceback
            raise CustomException(e, exc_traceback)  # Pass traceback object

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.load_data()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)


