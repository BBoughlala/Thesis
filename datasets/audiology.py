import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  train_data = pd.read_csv('datasets/audiology.standardized.data', header=None, na_values='?')
  test_data = pd.read_csv('datasets/audiology.standardized.test', header=None, na_values='?')
  data = pd.concat([train_data, test_data])
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])
  y = le.fit_transform(y)
  return X, y
