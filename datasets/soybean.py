from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  soybean_large = fetch_ucirepo(id=90) 
  X = soybean_large.data.features 
  y = soybean_large.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  y = le.fit_transform(y)
  return X, y