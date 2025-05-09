import pandas as pd
import numpy as np
import sys
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from src.exception import CustomException
from src.logger import logging


class DataPreprocessing:
    def __init__(self,train_data_path:str,test_data_path:str):
        try:
            self.train = pd.read_csv(train_data_path)
            self.test = pd.read_csv(test_data_path)
            logging.info('train and test data loaded')
        except Exception as e:
            raise CustomException(e,sys)
    
    def preprocessing(self):

        ## handling null values ##
        try:
            imputer = KNNImputer(n_neighbors=5)
            logging.info("KNN imputer initialized")

            train_imputed = imputer.fit_transform(self.train.drop(columns=['timestamp']))
            test_imputed = imputer.transform(self.test.drop(columns=['timestamp']))

            train_imputed = pd.DataFrame(train_imputed,columns=self.train.drop(columns=['timestamp']).columns)
            test_imputed = pd.DataFrame(test_imputed,columns=self.test.drop(columns=['timestamp']).columns)

            self.train = pd.concat((self.train[['timestamp']],train_imputed),axis=1)
            self.test = pd.concat((self.test[['timestamp']].reset_index(drop=True),test_imputed),axis=1)

            logging.info("null values replaced with KNN imputer")

            artifact_dir = r'DS-Intern-Assignment-Shivam-Ghuge\artifacts'
            os.makedirs(artifact_dir,exist_ok=True)
            with open(os.path.join(artifact_dir,'knn_imputer.pickle'),'wb') as f:
                pickle.dump(imputer,f)
            logging.info(f"knn imputer saved to the {os.path.split(artifact_dir)[-1]}")
        except Exception as e:
            raise CustomException(e,sys)
        

        ## Feature Scaling ##
        try:
            scaler = StandardScaler()

            train_scaled = scaler.fit_transform(self.train.drop(columns=['equipment_energy_consumption','timestamp']))
            test_scaled = scaler.transform(self.test.drop(columns=['equipment_energy_consumption','timestamp']))
            logging.info("train and test data scaled using StandardScaler")

            with open(os.path.join(artifact_dir,'scaler.pickle'),'wb') as f:
                pickle.dump(scaler,f)
            logging.info(f"standard scaler saved to the {os.path.split(artifact_dir)[-1]}")

        except Exception as e:
            raise CustomException(e,sys)
        
        try:
            train_scaled = pd.DataFrame(train_scaled,columns=self.train.drop(columns=['equipment_energy_consumption','timestamp']).columns)
            test_scaled = pd.DataFrame(test_scaled,columns=self.test.drop(columns=['equipment_energy_consumption','timestamp']).columns)

            train_scaled = pd.concat((self.train[['timestamp','equipment_energy_consumption']],train_scaled),axis=1)
            test_scaled = pd.concat((self.test[['timestamp','equipment_energy_consumption']],test_scaled),axis=1)

            logging.info("dataframe created with scaled values")
        except Exception as e:
            raise CustomException(e,sys)
        
        return train_scaled,test_scaled
    
    def export_data(self,train_data:pd.DataFrame,test_data:pd.DataFrame):
        try:
            directory = 'DS-Intern-Assignment-Shivam-Ghuge\data\preprocessed'
            os.makedirs(directory,exist_ok=True)
            train_data.to_csv(os.path.join(directory,'train.csv'),header=True,index=False)
            test_data.to_csv(os.path.join(directory,'test.csv'),header=True,index=False)

            logging.info(f"train and test data preprocessed and saved to {os.path.split(directory)[-1]} folder")
        
        except Exception as e:
            raise CustomException(e,sys)
    
