from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def fetch():
  chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22) 
  X = chess_king_rook_vs_king_pawn.data.features 
  y = chess_king_rook_vs_king_pawn.data.targets
  X = X.to_numpy()
  y = y.to_numpy()
  le = LabelEncoder()
  y = le.fit_transform(y)
  for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])
  X = X.astype(np.float64)
  y = y.astype(np.float64)
  return X, y