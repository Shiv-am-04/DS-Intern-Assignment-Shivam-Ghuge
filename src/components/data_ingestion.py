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
            
            # converting the datatype of some of features
            col = [col for col in data.columns if data[col].dtype == 'object']
            col.remove('timestamp')

            for i in col:
                self.df[i] = pd.to_numeric(self.df[i],errors='coerce')
                
        except Exception as e:
            raise CustomException(e,sys)

    def split_data(self,test_size=0.3):

        try:
            self.df.sort_values(by='timestamp',inplace=True)
            train,test = train_test_split(self.df,test_size=test_size,shuffle=False,random_state=0)
            logging.info(f"data splitted into train and test : [train_size:{train.shape}, test_size:{test.shape}]")
        except Exception as e:
            raise CustomException(e,sys)

        return train,test
    
    def export_data(self,train:pd.DataFrame,test:pd.DataFrame):

        try:
            path = 'DS-Intern-Assignment-Shivam-Ghuge\data\splited'
            os.makedirs(path,exist_ok=True)
            train.to_csv(os.path.join(path,'train.csv'),index=False)
            test.to_csv(os.path.join(path,'test.csv'),index=False)
            logging.info(f"train and test data saved to {os.path.split(path)[-1]} folder")
        except Exception as e:
            raise CustomException(e,sys)
