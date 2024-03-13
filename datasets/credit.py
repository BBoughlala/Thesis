from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  cred = fetch_ucirepo(id=27)
  X = cred.data.features
  y = cred.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  for i in [2, 3, 5, 6, 8, 9, 10, 11, 14]:
    X[:, i] = le.fit_transform(X[:, i])
  y = le.fit_transform(y)
  X = X.astype(float)
  y = y.astype(float)
  return X, y