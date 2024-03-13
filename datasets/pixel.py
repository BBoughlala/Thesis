import pandas as pd 
import numpy as np

def fetch():
  X = pd.read_csv(r'datasets\mfeat-pix', delim_whitespace=True, header=None)
  classes = np.arange(1, 10)
  y = np.repeat(classes, 200)
  return X, y