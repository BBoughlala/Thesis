from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  wisconsin = fetch_ucirepo(id=15) 
  X = wisconsin.data.features 
  y = wisconsin.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y