from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  breast_cancer = fetch_ucirepo(id=14) 
  X = breast_cancer.data.features 
  y = breast_cancer.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])
  y = le.fit_transform(y)
  return X, y

