import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print(sys.path)
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self):
        try:
            logging.info("Loading data from raw data path...")
            df = pd.read_csv('data/data.csv')

            logging.info('Reading data from raw data path is successful!')
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated!')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is successful!")

            return(
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.load_data()



