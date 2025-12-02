import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, Y_train)

print("The best parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
Y_pred_rf = best_model.predict(X_test)

mae = mean_absolute_error(y_true = Y_test, y_pred = Y_pred_rf)
mse = mean_squared_error(y_true = Y_test, y_pred = Y_pred_rf)
rmse = mse ** 0.5
r2 = r2_score(y_true = Y_test, y_pred = Y_pred_rf)

print("MAE of Random Forest Regressor Algorithm = ", mae)
print("RMSE of Random Forest Regressor Algorithm = ", rmse)
print("R^2 of Random Forest Regressor Algorithm = ", r2)