from src.exception import CustomException
from src.logger import logging
import sys
import os
import pickle
import json

from lightgbm import LGBMRegressor


class ModelTraining:
    def __init__(self,X_train,X_test,y_train,y_test):
        try:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        except Exception as e:
            raise CustomException(e,sys)
        

    def model_training(self):
        try:
            params = {
                'boosting_type' : 'rf',
                'max_depth' : 8,
                'num_leaves' : 15,
                'learning_rate' : 0.01,
                'n_estimators' : 300,
                'reg_alpha' : 1,
                'reg_lambda' : 1,
                'importance_type' : 'gain',
                'bagging_fraction' : 0.6,  
                'feature_fraction' : 0.7
            }

            model = LGBMRegressor(
                boosting_type = 'rf',
                max_depth = 8,
                num_leaves = 15,
                learning_rate = 0.01,
                n_estimators = 300,
                reg_alpha = 1,
                reg_lambda = 1,
                importance_type = 'gain',
                bagging_fraction = 0.6,  
                feature_fraction = 0.7
                )

            logging.info(f"model initialized with params : {params}")

            artifact_dir = r'DS-Intern-Assignment-Shivam-Ghuge\artifacts'
            os.makedirs(artifact_dir,exist_ok=True)
            with open(os.path.join(artifact_dir,'params.json'),'w') as f:
                json.dump(params,f,indent=4)
                
        
        except Exception as e:
            raise CustomException(e,sys)
        
        try:
            model.fit(self.X_train,self.y_train)
            logging.info("model fitted successfully")
        except Exception as e:
            raise CustomException(e,sys)
        
        return model

    def save_model(self,model_trained):
        try:
            directory = r'DS-Intern-Assignment-Shivam-Ghuge\models'
            os.makedirs(directory,exist_ok=True)

            with open(os.path.join(directory,'lgbm_model.pickle'),'wb') as f:
                pickle.dump(model_trained,f)
            logging.info(f"model saved to {os.path.split(directory)[-1]} folder")
        except Exception as e:
            raise CustomException(e,sys)
        