import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  data = pd.read_csv(r'C:\Users\Bilal\Desktop\TUe\Y2\Q2\code\Thesis\datasets\sponge.data', header=None, na_values='?')
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  y = le.fit_transform(y)
  for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])
  X = X.astype(np.float32)
  y = y.astype(np.float32)
  return X, y
