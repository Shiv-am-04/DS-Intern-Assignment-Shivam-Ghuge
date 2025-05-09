import pandas as pd
import numpy as np
import sys
import os

from scipy.stats import zscore

from src.exception import CustomException
from src.logger import logging


class AnomalyDetection:
    def __init__(self,train:pd.DataFrame,test:pd.DataFrame):
        try:
            self.train = train
            self.test = test
        except Exception as e:
            raise CustomException(e,sys)
        
    def detect_and_remove_outliers(self,method='iqr'):
        """
        Detects and removes outliers from the DataFrame.
        
        Parameters:
        - method (str): Method for anomaly detection. Options are:
            - 'zscore'
            - 'iqr'
        """

        if method == 'zscore':
            # Z-Score method
            z_scores = np.abs(zscore(self.train.select_dtypes(include=[np.number])))
            self.train = self.train[(z_scores < 3).all(axis=1)]

        elif method == 'iqr':
            # IQR method
            Q1 = self.train.quantile(0.25)
            Q3 = self.train.quantile(0.75)
            IQR = Q3 - Q1
            self.train = self.train[~((self.train < (Q1 - 1.5 * IQR)) | (self.train > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # Reset index to maintain alignment
        self.train.reset_index(drop=True, inplace=True)
        logging.info(f"Outliers removed. Original rows: {len(self.train)}, Cleaned rows: {len(self.train)}")


    def export_data(self):
        try:
            directory = 'DS-Intern-Assignment-Shivam-Ghuge\data\clean_data'
            os.makedirs(directory,exist_ok=True)

            X_train = self.train.drop(columns=['equipment_energy_consumption','random_variable1','random_variable2'])
            X_test = self.test.drop(columns=['equipment_energy_consumption','random_variable1','random_variable2'])

            y_train = self.train['equipment_energy_consumption']
            y_test = self.test['equipment_energy_consumption']

            X_train.to_csv(os.path.join(directory,'X_train.csv'),index=False)
            X_test.to_csv(os.path.join(directory,'X_test.csv'),index=False)

            y_train.to_csv(os.path.join(directory,'y_train.csv'),index=False)
            y_test.to_csv(os.path.join(directory,'y_test.csv'),index=False)
            logging.info(f"X_train,X_test,y_train and y_test data saved to {os.path.split(directory)[-1]} folder")
        except Exception as e:
            raise CustomException(e,sys)    
    
        return X_test,X_test,y_train,y_test

