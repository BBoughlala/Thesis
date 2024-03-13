import numpy as np
import pandas as pd

def fetch():
  data = pd.read_csv(r'C:\Users\Bilal\Desktop\TUe\Y2\Q2\code\Thesis\datasets\waveform.data', header=None, na_values='?')
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  X = X.to_numpy()
  y = y.to_numpy()
  return X, y