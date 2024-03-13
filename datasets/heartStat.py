from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  statlog_heart = fetch_ucirepo(id=145) 
  X = statlog_heart.data.features 
  y = statlog_heart.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y