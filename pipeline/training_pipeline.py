import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.feature_engineering import FeatureEngineering
from src.components.anomaly_detection import AnomalyDetection
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation


# reading raw data
data = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\data.csv')

### Data Ingestion ### 

ingestion = DataIngestion(data=data)

train,test = ingestion.split_data()
ingestion.export_data(train,test)


## Data Prerocessing ##

preprocessor = DataPreprocessing(train,test)

train_scaled,test_scaled = preprocessor.preprocessing()

preprocessor.export_data(train_scaled,test_scaled)


## Feature engineering ## 

fe = FeatureEngineering(train_scaled,test_scaled)
fe.feature_handeling()
train_data,test_data = fe.encoding_feature()
fe.export_data(train_data,test_data)

## Anomaly Detection ##

ad = AnomalyDetection(train_data,test_data)
ad.detect_and_remove_outliers()
X_train,X_test,y_train,y_test = ad.export_data()

## Model training ##

mt = ModelTraining(X_train,X_test,y_train,y_test)
model = mt.model_training()
mt.save_model(model_trained=model)
