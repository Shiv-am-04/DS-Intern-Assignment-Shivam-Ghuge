from src.components.model_evaluation import ModelEvaluation
import pickle
import pandas as pd


model = pickle.load(r'DS-Intern-Assignment-Shivam-Ghuge\artifact\lgbm_model.pickle')

X_test = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\clean_data\X_test.csv')
y_test = pd.read_csv(r'DS-Intern-Assignment-Shivam-Ghuge\data\clean_data\y_test.csv')

## Model Evaluation ##

evaluate = ModelEvaluation(model,X_test,y_test)
y_pred = evaluate.prediction()
evaluate.evaluation(y_pred)