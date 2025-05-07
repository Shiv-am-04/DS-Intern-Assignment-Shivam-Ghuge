import pandas as pd
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self,data:pd.DataFrame):

        try:
            self.df = data
            column = "timestamp"
            self.df.drop(columns=column,inplace=True)
            logging.info(f"{column} column removed successfully")
        except Exception as e:
            raise CustomException(e,sys)

    def split_data(self,test_size=0.3):

        try:
            train,test = train_test_split(self.df,test_size=test_size,random_state=0)
            logging.info(f"data splitted into train and test : [train_size:{train.shape}, test_size:{test.shape}]")
        except Exception as e:
            raise CustomException(e,sys)

        return train,test
    
    def export_data(self,train:pd.DataFrame,test:pd.DataFrame):

        try:
            path = 'DS-Intern-Assignment-Shivam-Ghuge\data\processed'
            os.makedirs(path,exist_ok=True)
            train.to_csv(os.path.join(path,'train.csv'))
            test.to_csv(os.path.join(path,'test.csv'))
            logging.info(f"train and test data saved to {os.path.split(path)[-1]} folder")
        except Exception as e:
            raise CustomException(e,sys)
