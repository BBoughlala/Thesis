from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  statlog_vehicle_silhouettes = fetch_ucirepo(id=149) 
  X = statlog_vehicle_silhouettes.data.features 
  y = statlog_vehicle_silhouettes.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  y = le.fit_transform(y)
  return X, y