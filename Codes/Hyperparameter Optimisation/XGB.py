import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

xgb = xg.XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
}

grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train)

print("The best parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
Y_pred_xgb = best_model.predict(X_test)

mae = mean_absolute_error(y_true = Y_test, y_pred = Y_pred_xgb)
rmse = mean_squared_error(y_true = Y_test, y_pred = Y_pred_xgb)
rmse = rmse ** 0.5
r2 = r2_score(y_true = Y_test, y_pred = Y_pred_xgb)

print("MAE of XGboost Regressor Algorithm = ", mae)
print("RMSE of XGboost Regressor Algorithm = ", rmse)
print("R^2 of XGboost Regressor Algorithm = ", r2)