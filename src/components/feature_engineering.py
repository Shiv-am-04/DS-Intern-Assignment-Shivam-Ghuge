import pandas as pd
import numpy as np
import sys
import os

from src.exception import CustomException
from src.logger import logging


class FeatureEngineering:
    def __init__(self,train_data:pd.DataFrame,test_data:pd.DataFrame):

        try:
            self.train = train_data
            self.test = test_data
        except Exception as e:
            raise CustomException(e,sys)
        
    def feature_handeling(self):
        # converting to datetime datatype because the datatype of timestamp feature is object
        try:
            self.train['timestamp'] = pd.to_datetime(self.train['timestamp'])
            self.test['timestamp'] = pd.to_datetime(self.test['timestamp'])
            logging.info("converted 'timestamp' feature to datetime dataype")

            # creating new features
            self.train['hour'] = self.train['timestamp'].dt.hour
            self.train['minute'] = self.train['timestamp'].dt.minute
            self.train['month'] = self.train['timestamp'].dt.month
            self.train['day_of_week'] = self.train['timestamp'].dt.dayofweek

            self.test['hour'] = self.test['timestamp'].dt.hour
            self.test['minute'] = self.test['timestamp'].dt.minute
            self.test['month'] = self.test['timestamp'].dt.month
            self.test['day_of_week'] = self.test['timestamp'].dt.dayofweek
            logging.info("new features : [hour,minute,day_of_week,month] created using 'timestamp' feature for both train and test dataset")
        except Exception as e:
            raise CustomException(e,sys)

        
    def encoding_feature(self):
        try:
            self.train['hour_sin'] = np.sin(2 * np.pi * self.train['hour'] / 24)
            self.train['hour_cos'] = np.cos(2 * np.pi * self.train['hour'] / 24)

            # Minute of the hour (0 to 59)
            self.train['minute_sin'] = np.sin(2 * np.pi * self.train['minute'] / 60)
            self.train['minute_cos'] = np.cos(2 * np.pi * self.train['minute'] / 60)

            # Day of the week (0 to 6)
            self.train['day_of_week_sin'] = np.sin(2 * np.pi * self.train['day_of_week'] / 7)
            self.train['day_of_week_cos'] = np.cos(2 * np.pi * self.train['day_of_week'] / 7)

            # Month of the year (1 to 12)
            self.train['month_sin'] = np.sin(2 * np.pi * (self.train['month'] - 1) / 12)
            self.train['month_cos'] = np.cos(2 * np.pi * (self.train['month'] - 1) / 12)
            logging.info("encoded train data")

            self.train.drop(columns=['timestamp','hour','minute','day_of_week','month'],inplace=True)
            logging.info("columns : ['timestamp','hour','minute','day_of_week','month'] droped from train dataset")
        except Exception as e:
            raise CustomException(e,sys)

        try:
            # for test dataset
            self.test['hour_sin'] = np.sin(2 * np.pi * self.test['hour'] / 24)
            self.test['hour_cos'] = np.cos(2 * np.pi * self.test['hour'] / 24)

            # Minute of the hour (0 to 59)
            self.test['minute_sin'] = np.sin(2 * np.pi * self.test['minute'] / 60)
            self.test['minute_cos'] = np.cos(2 * np.pi * self.test['minute'] / 60)

            # Day of the week (0 to 6)
            self.test['day_of_week_sin'] = np.sin(2 * np.pi * self.test['day_of_week'] / 7)
            self.test['day_of_week_cos'] = np.cos(2 * np.pi * self.test['day_of_week'] / 7)

            # Month of the year (1 to 12)
            self.test['month_sin'] = np.sin(2 * np.pi * (self.test['month'] - 1) / 12)
            self.test['month_cos'] = np.cos(2 * np.pi * (self.test['month'] - 1) / 12)
            logging.info("encoded test data")

            self.test.drop(columns=['timestamp','hour','minute','day_of_week','month'],inplace=True)
            logging.info("columns : ['timestamp','hour','minute','day_of_week','month'] droped from test dataset")
        except Exception as e:
            raise CustomException(e,sys)

        return self.train,self.test
        
    def export_data(self,train:pd.DataFrame,test:pd.DataFrame):
        try:
            path = 'DS-Intern-Assignment-Shivam-Ghuge\data\FE_data'
            os.makedirs(path,exist_ok=True)
            train.to_csv(os.path.join(path,'train.csv'),index=False)
            test.to_csv(os.path.join(path,'test.csv'),index=False)
            logging.info(f"train and test data saved to {os.path.split(path)[-1]} folder")
        except Exception as e:
            raise CustomException(e,sys)