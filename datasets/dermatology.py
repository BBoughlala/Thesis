from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  dermatology = fetch_ucirepo(id=33)
  X = dermatology.data.features
  y = dermatology.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y