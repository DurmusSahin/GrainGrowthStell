import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

from sklearn.svm import SVR
svr = SVR()
param_grid = {
    'C': [0.1, 1, 2, 3, 4, 5, 10, 100],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'] 
}

grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)

print(f"The best parameters: {grid_search.best_params_}")

best_svr = grid_search.best_estimator_

Y_pred_svr = best_svr.predict(X_test)

mae = mean_absolute_error(y_true = Y_test, y_pred = Y_pred_svr)
mse = mean_squared_error(y_true = Y_test, y_pred = Y_pred_svr)
rmse = mse ** 0.5
r2 = r2_score(y_true = Y_test, y_pred = Y_pred_svr)

print("MAE of Support Vector Regression Algorithm = ", mae)
print("RMSE of Support Vector Regression Algorithm = ", rmse)
print("R^2 of Support Vector Regression Algorithm = ", r2)