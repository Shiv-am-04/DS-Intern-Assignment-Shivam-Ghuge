## Approach to the problem
- First I have find the NaN values and removed them using KNN imputer.
- After that I have start data preprocessing like scaling the features. I have scaled using sklearn's standard scaler.
- There is a feature name timestamp which is important because energy consumption is calculated over time, just blindly removing it creates problem because it will be useful for the model to learn temporal dependancy.
- We cannot use datetime data directly to the model that's why I have created new features form the timestamp feature which are month,day of week,hour and minute.
- Since month,week,hour and minute are cyclical as they repeat, therefore I used sin,cos cylical encoding because without it model doesn't understand that 23:59 and 1:00 are very close to each other.
- As stated in the description , I performed some staistical analysis like variance inflation factor, mutual information score and correlation with target on random_variable1 and random_variable2 to know whether to include them or not in dataset.
- After that I have removed the outliers with the help Inter Quartile Range (IQR).
- I also calculated the feature importance of each feature using xgboost model. There I got know that majority of the features have very low influence towards the target variable resulting in very low r2 score.

## Key insights from the data
- Features have very less influence on the target variable.
- timestamp proves to be the important feature.
- Features have very low or no correlation with each other which is good. But they also have very low correlation with target which is bad.

## Model performance evaluation
- Model is evaluated on three metrics Root mean squared error, mean absolute error and R2 score.
- R2 score is very low due to the very less influence of features towards dataset
- RMSE is less than 10% of the prediction range and MAE is nearly 8% of the prediction rate
- Overall model provide very less error except some where error is very high.
- 
## Recommendations for reducing equipment energy consumption
- From the model and the features that influence the most for predicting target is machine operated in hours, atmospheric pressure, temprature and humidity of some zones 
- This shows that using machinery by taking some break if it operates for continous hours can help the machinery to consume optimal energy just like humans after taking rest our productivity improves.
- Outdoor temprature and humidity, wind speed, visibilty has nearly no influence.
