from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  horse = fetch_ucirepo(id=47)
  X = horse.data.features
  y = horse.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y