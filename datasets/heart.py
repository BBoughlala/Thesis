from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  heart = fetch_ucirepo(id=45)
  X = heart.data.features
  y = heart.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y