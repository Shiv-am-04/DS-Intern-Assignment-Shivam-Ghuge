from src.exception import CustomException
from src.logger import logging
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
import pandas as pd

# checking error handling and logging

# try:
#     a = 1/0
#     logging.info('division done correctly')
# except Exception as e:
#     raise CustomException(e,sys) from e

# checking data ingestion

# data = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\data.csv')

# ingestion = DataIngestion(data=data)

# train,test = ingestion.split_data()
# ingestion.export_data(train,test)

# # checking data prerocessing

# train_path = r'DS-Intern-Assignment-Shivam-Ghuge\data\splited\train.csv'
# test_path = r'DS-Intern-Assignment-Shivam-Ghuge\data\splited\test.csv'

# preprocessor = DataPreprocessing(train_path,test_path)

# train_scaled,test_scaled = preprocessor.preprocessing()

# preprocessor.export_data(train_scaled,test_scaled)

# checking feature_engineering

train_ = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\preprocessed\train.csv')
test_ = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\preprocessed\test.csv')

fe = FeatureEngineering(train_,test_)
fe.feature_handeling()
train_data,test_data = fe.encoding_feature()
fe.export_data(train_data,test_data)