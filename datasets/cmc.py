from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def fetch():
  contraceptive_method_choice = fetch_ucirepo(id=30) 
  X = contraceptive_method_choice.data.features
  y = contraceptive_method_choice.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y