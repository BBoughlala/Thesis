from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  optical = fetch_ucirepo(id=80)
  X = optical.data.features
  y = optical.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y