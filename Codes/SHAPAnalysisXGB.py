import shap
import xgboost
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("STEELDATA.xlsx")
X = df.drop(columns=["D"])
Y = df["D"]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 5)

import xgboost as xg
model = xg.XGBRegressor(colsample_bytree = 0.9, gamma = 0.2, learning_rate = 0.1, max_depth = 5, n_estimators = 300, subsample = 0.8)
model.fit(X_train, Y_train)

explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=15)
shap.plots.beeswarm(shap_values, max_display=15)
plt.show()