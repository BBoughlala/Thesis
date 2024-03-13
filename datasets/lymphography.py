import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  data = pd.read_csv(r'datasets\lymphography.data', header=None, na_values='?')
  y = data.iloc[:, 0]
  X = data.iloc[:, 1:]
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y