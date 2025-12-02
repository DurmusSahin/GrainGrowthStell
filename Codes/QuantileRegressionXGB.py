from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_pinball_loss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_excel('steel-data.xlsx')
X = dataset.iloc[:, 0:24].values
Y = dataset.iloc[:, 24].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)


# Define the quantiles for the prediction intervals
quantiles = [0.025, 0.5, 0.975]

# Train an XGBRegressor for each quantile
models = {}
for quantile in quantiles:
    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=quantile, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    models[quantile] = model

# Make predictions on the test set for each quantile
predictions = {}
for quantile in quantiles:
    predictions[quantile] = models[quantile].predict(X_test)

# Calculate the pinball loss for each quantile
for quantile in quantiles:
    quantile_loss = mean_pinball_loss(y_test, predictions[quantile], alpha=quantile)
    print(f"Pinball Loss at {quantile} quantile: {quantile_loss:.4f}")

# Visualize the predicted intervals and true values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="True Values")
plt.scatter(range(len(predictions[0.5])), predictions[0.5], color="red", label="Mean Values")
plt.fill_between(range(len(y_test)), predictions[0.025], predictions[0.975], color="red", alpha=0.2, label="95% Prediction Interval")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.title("Prediction Intervals with XGBoost Quantile Regression")
plt.legend()
plt.show()