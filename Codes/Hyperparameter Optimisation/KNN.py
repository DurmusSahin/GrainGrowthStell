import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()


param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]                  
}

grid_search = GridSearchCV(estimator=knn_reg, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)

print("The best parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
Y_pred_knn = best_model.predict(X_test)

mae = mean_absolute_error(y_true = Y_test, y_pred = Y_pred_knn)
mse = mean_squared_error(y_true = Y_test, y_pred = Y_pred_knn)
rmse = mse ** 0.5
r2 = r2_score(y_true = Y_test, y_pred = Y_pred_knn)

print("MAE of KNN Regression Algorithm = ", mae)
print("RMSE of KNN Regression Algorithm = ", rmse)
print("R^2 of KNN Regression Algorithm = ", r2)