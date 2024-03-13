from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  zoo = fetch_ucirepo(id=111) 
  X = zoo.data.features 
  y = zoo.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y