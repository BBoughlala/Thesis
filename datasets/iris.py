from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  iris = fetch_ucirepo(id=53)
  X = iris.data.features
  y = iris.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  y = le.fit_transform(y)
  return X, y