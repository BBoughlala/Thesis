from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  hepatitis = fetch_ucirepo(id=46)
  X = hepatitis.data.features
  y = hepatitis.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y