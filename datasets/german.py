from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  statlog_german_credit_data = fetch_ucirepo(id=144)
  X = statlog_german_credit_data.data.features
  y = statlog_german_credit_data.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
    X[:, i] = le.fit_transform(X[:, i])
  X = X.astype(np.float64)
  return X, y
