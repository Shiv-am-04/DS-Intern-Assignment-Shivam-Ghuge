from src.exception import CustomException
from src.logger import logging
import sys

from src.components.data_ingestion import DataIngestion
import pandas as pd

# checking error handling and logging

# try:
#     a = 1/0
#     logging.info('division done correctly')
# except Exception as e:
#     raise CustomException(e,sys) from e

# checking data ingestion

data = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\data.csv')

ingestion = DataIngestion(data=data)

train,test = ingestion.split_data()
ingestion.export_data(train,test)
