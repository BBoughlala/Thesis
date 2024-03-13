import pandas as pd
import numpy as np

def fetch():
  data = pd.read_csv('datasets/arrhythmia.data', header=None, na_values='?')
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y