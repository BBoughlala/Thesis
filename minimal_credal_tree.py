from minimal_credal_joint import CredalJoint
import numpy as np
import time

class Node():
      def __init__(self) -> None:
            self.feature = None
            self.value = None
            self.left = None
            self.right = None
            self.label = None

class DecisionTree():
      def __init__(self, s:int, maxiter:int, base:int) -> None:
            self.s = s
            self.maxiter = maxiter
            self.base = base
            self.tree = None

      def partition(self, X: np.ndarray, y: np.ndarray, feature: int, value: int) -> np.ndarray:
            X_missing_mask = np.isnan(X[:, feature])
            
            X_complete = X[~X_missing_mask]
            X_missing = X[X_missing_mask]
            y_complete = y[~X_missing_mask]
            y_missing = y[X_missing_mask]
            
            left_mask = X_complete[:, feature] <= value
            
            left_X = X_complete[left_mask]
            right_X = X_complete[~left_mask]
            left_y = y_complete[left_mask]
            right_y = y_complete[~left_mask]
            
            left_X = np.concatenate((left_X, X_missing), axis=0)
            right_X = np.concatenate((right_X, X_missing), axis=0)
            left_y = np.concatenate((left_y, y_missing), axis=0)
            right_y = np.concatenate((right_y, y_missing), axis=0)
            
            return left_X, right_X, left_y, right_y
      
      def majority_class(self, y:np.ndarray) -> int:
            y_copy = y.copy()
            y_copy = y_copy[~np.isnan(y_copy)]
            y_copy = y_copy.reshape(-1)
            counts = np.bincount(y_copy.astype(int))
            return np.argmax(counts)

      def build_tree(self, X:np.ndarray, y:np.ndarray, root:Node=None, depth:int=0) -> Node:
            root = Node() if root is None else root
            if depth >= 100:
                  root.label = self.majority_class(y)
                  return root
            elif len(np.unique(y[~np.isnan(y)])) == 1:
                  root.label = self.majority_class(y)
                  return root
            else:
                  best_lb = -np.inf
                  best_ub = -np.inf
                  best_feature = []
                  y = y.reshape(-1,1)
                  idx = np.arange(len(y)).reshape(-1,1)
                  idx_y = np.hstack( (idx, y) )
                  idx_y = idx_y[idx_y[:,1].argsort()]
                  thresholds_idx = []
                  values = np.unique(y[~np.isnan(y)]).astype(int)
                  for v in values:
                        idx_v = np.where(idx_y[:,1] == v)[0][0]
                        idx_t = idx_y[idx_v, 0]
                        thresholds_idx.append(idx_t)
                  for feature in range(X.shape[1]):
                        for idx in thresholds_idx:
                              idx = int(idx)
                              v = X[idx, feature]
                              credal_joint = CredalJoint(self.s, self.maxiter, self.base)
                              credal_joint.fit(X[:,feature], y, v)
                              lb, ub = credal_joint.information_gain(0)
                              if lb > best_ub:
                                    best_lb = lb
                                    best_ub = ub
                                    best_feature = [(feature, v, lb, ub)]
                              elif lb > best_lb:
                                    best_lb = lb
                                    best_feature.append((feature, v, lb, ub))
                                    best_feature = [i for i in best_feature if i[3] >= best_lb]
                              else:
                                    continue
                  best_feature = [] if best_lb == 0 else best_feature
                  if len(best_feature) > 1:
                        best_feature = max(best_feature, key=lambda x: x[2])
                        root.feature = best_feature[0]
                        root.value = best_feature[1]
                        left_X, right_X, left_y, right_y = self.partition(X, y, root.feature, root.value)
                        if len(left_y[~np.isnan(left_y)]) == 0 or len(right_y[~np.isnan(right_y)]) == 0:
                              root.label = self.majority_class(y)
                              return root
                        else:
                              left_child = Node()
                              right_child = Node()
                              root.left = self.build_tree(left_X, left_y, left_child, depth+1)
                              root.right = self.build_tree(right_X, right_y, right_child, depth+1)
                  elif len(best_feature) == 0:
                        root.label = self.majority_class(y)
                        return root
                  else:
                        root.feature = best_feature[0][0]
                        root.value = best_feature[0][1]
                        left_X, right_X, left_y, right_y = self.partition(X, y, root.feature, root.value)
                        if len(left_y[~np.isnan(left_y)]) == 0 or len(right_y[~np.isnan(right_y)]) == 0:
                              root.label = self.majority_class(y)
                              return root
                        else:
                              left_child = Node()
                              right_child = Node()
                              root.left = self.build_tree(left_X, left_y, left_child, depth+1)
                              root.right = self.build_tree(right_X, right_y, right_child, depth+1)
            return root

      def fit(self, X:np.ndarray, y:np.ndarray) -> None:
            self.tree = self.build_tree(X, y)
      
      def predict(self, X:np.ndarray) -> np.ndarray:
            y_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                  node = self.tree
                  while node.label is None:
                        if X[i,node.feature] <= node.value:
                              node = node.left
                        else:
                              node = node.right
                  y_pred[i] = node.label
            return y_pred
      
      def get_depth(self, node:Node=None) -> int:
            if node is None:
                  node = self.tree
            if node.label is not None:
                  return 1
            else:
                  return 1 + max(self.get_depth(node.left), self.get_depth(node.right))