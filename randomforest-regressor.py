import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


#Load data
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target 

#Split Model
X = df.drop(['MEDV'], axis = 1) 
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

#Model Creation
regressor = RandomForestRegressor(n_estimators=30) 
regressor.fit(X_train, y_train) 

#Model prediction
Y_Pred = regressor.predict(X_test)
MAE = metrics.mean_absolute_error(y_test, Y_Pred) 
RMSE = np.sqrt(metrics.mean_squared_error(y_test, Y_Pred))
print('RMSE: ', RMSE)