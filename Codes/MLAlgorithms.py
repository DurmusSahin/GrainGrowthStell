import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)


from sklearn.ensemble import AdaBoostRegressor
adaboost_reg = AdaBoostRegressor(learning_rate = 0.01, n_estimators = 225)
adaboost_reg.fit(X_train, Y_train)
Y_pred_adaboost = adaboost_reg.predict(X_test)
adaboost_errors = Y_test - Y_pred_adaboost

from sklearn import tree
dt_reg = tree.DecisionTreeRegressor(max_depth = None, min_samples_leaf = 1, min_samples_split = 2, random_state=42)
dt_reg.fit(X_train, Y_train)
Y_pred_dt = dt_reg.predict(X_test)
dt_errors = Y_test - Y_pred_dt

from sklearn.gaussian_process import GaussianProcessRegressor
gaussian_process = GaussianProcessRegressor()
gaussian_process.fit(X_train, Y_train)
Y_pred_gp = gaussian_process.predict(X_test)
gpr_errors = Y_test - Y_pred_gp

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(algorithm = 'auto', n_neighbors = 1, p = 1, weights = 'uniform')
knn_reg.fit(X_train, Y_train)
Y_pred_knn = knn_reg.predict(X_test)
knn_errors = Y_test - Y_pred_knn

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
lr_errors = Y_test - Y_pred_lr

from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(activation = 'relu', alpha = 0.001, hidden_layer_sizes = (100, 100), learning_rate = 'constant', solver = 'lbfgs')
mlp_regr.fit(X_train, Y_train)
Y_pred_mlp = mlp_regr.predict(X_test)
mlp_errors = Y_test - Y_pred_mlp

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(bootstrap = True, max_depth = 20, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 300)
rf_reg.fit(X_train, Y_train)
Y_pred_rf = rf_reg.predict(X_test)
rf_errors = Y_test - Y_pred_rf

from sklearn.svm import SVR
svr = SVR(kernel='linear',C=1.5,epsilon=0.01,gamma='scale')
svr.fit(X_train, Y_train)
Y_pred_svr = svr.predict(X_test)
svr_errors = Y_test - Y_pred_svr

import xgboost as xg
xgb_r = xg.XGBRegressor(colsample_bytree = 0.9, gamma = 0.2, learning_rate = 0.1, max_depth = 5, n_estimators = 300, subsample = 0.8)
xgb_r.fit(X_train, Y_train)
Y_pred_xgb = xgb_r.predict(X_test)
xgb_errors = Y_test - Y_pred_xgb

all_errors = np.concatenate([adaboost_errors, dt_errors, gpr_errors, knn_errors, lr_errors, mlp_errors, rf_errors, svr_errors, xgb_errors])
model_names = ['Adaboost'] * 208 + ['DT'] * 208 + ['GPR'] * 208 + ['KNN'] * 208 + ['LR'] * 208 + ['MLP'] * 208 + ['RF'] * 208 + ['SVR'] * 208 + ['XGBoost'] * 208

print(all_errors.shape)
print(len(model_names))

data = {'ML Algorithms': model_names, 'Absolute Error': np.abs(all_errors)}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.boxplot(x='ML Algorithms', y='Absolute Error', data=df)
plt.title('Absolute Errors of ML Algorithms')
plt.show()