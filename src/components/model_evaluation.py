from src.exception import CustomException
from src.logger import logging
import sys
import os
import numpy as np
import pickle


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

class ModelEvaluation:
    def __init__(self,model,X_test,y_test):
        try:
            self.model = model
            self.X_test = X_test
            self.y_test = y_test
        except Exception as e:
            raise CustomException(e,sys)
    
    def prediction(self):
        try:
            y_pred = self.model.predict(self.X_test)
            logging.info("prediction done")

        except Exception as e:
            raise CustomException(e,sys)

        return y_pred
    
    def evaluation(self,y_pred):
        try:
            rmse = np.sqrt(mean_squared_error(self.y_test,y_pred))
            mae = mean_absolute_error(self.y_test,y_pred)
            r2 = r2_score(self.y_test,y_pred)
            logging.info(f"metrics calculated : ['rmse':{rmse},'mae':{mae},'r2':{r2}]")

            metrics_dir = r'DS-Intern-Assignment-Shivam-Ghuge\results'
            os.makedirs(metrics_dir,exist_ok=True)

            with open(os.path.join(metrics_dir,'metrics.txt'),'w') as f:
                f.write(f"root_mean_squared_error : {rmse}")
                f.write("\n") 
                f.write(f"mean_absolute_error : {mae}") 
            logging.info(f"metric saved to the {os.path.split(metrics_dir)[-1]}")
        except Exception as e:
            raise CustomException(e,sys)
    

