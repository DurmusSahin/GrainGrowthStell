import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('STEELDATA.xlsx')
features = df.iloc[:, :-1]
target = df.iloc[:, -1]
scaler = MinMaxScaler()
yeni = scaler.fit_transform(features)
dfyeni = pd.DataFrame(yeni)
dfyeni = dfyeni.join(target)
dfyeni.to_excel("steel-data.xlsx")