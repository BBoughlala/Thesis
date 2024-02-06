import numpy as np
from impreciseInformationGain import imprecise_information_gain
from copy import deepcopy

def best_split(x, y):
    """
    Find the best split threshold for a given attribute.

    Parameters:
    - x (numpy.ndarray): The attribute values.
    - y (numpy.ndarray): The target labels.

    Returns:
    - best_threshold: The best split threshold.
    - best_iig: The corresponding imprecise information gain.
    """
    best_threshold = None
    best_iig = -float('inf')
    unique_x = np.unique(x)
    for x_val in unique_x:
        iig = imprecise_information_gain(y, x_val, x, 1)
        if iig > best_iig:
            best_iig = iig
            best_threshold = x_val
    return best_threshold, best_iig

def find_best_attribute_split(X, y):
    """
    Find the best attribute and split threshold for the dataset.

    Parameters:
    - X (numpy.ndarray): The input features.
    - y (numpy.ndarray): The target labels.

    Returns:
    - best_attr: Index of the best attribute.
    - best_threshold: The best split threshold.
    - best_iig: The corresponding imprecise information gain.
    """
    n_attributes = X.shape[1]
    best_attr = None
    best_threshold = None
    best_iig = -float('inf')
    for i in range(n_attributes):
        threshold, iig = best_split(X[:, i], y)
        if iig > best_iig:
            best_iig = iig
            best_threshold = threshold
            best_attr = i
    return best_attr, best_threshold, best_iig

def partition(X, y, attribute, threshold):
    """
    Split the dataset based on a given attribute and threshold.

    Parameters:
    - X (numpy.ndarray): The input features.
    - y (numpy.ndarray): The target labels.
    - attribute (int): Index of the attribute to split on.
    - threshold: The split threshold.

    Returns:
    - X_left, y_left: Data on the left side of the split.
    - X_right, y_right: Data on the right side of the split.
    """
    left_mask = X[:, attribute] == threshold
    right_mask = X[:, attribute] != threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def most_common_value(arr):
    """
    Find the most common value in an array.

    Parameters:
    - arr (numpy.ndarray): The input array.

    Returns:
    - most_common_value: The most common value.
    """
    flattened_arr = np.asarray(arr).flatten()
    flattened_arr = flattened_arr.astype(int)	
    
    # Check if the flattened array is empty
    if len(flattened_arr) == 0:
        return None
    
    counts = np.bincount(flattened_arr)
    most_common_value = np.argmax(counts)
    
    return most_common_value

def leaf_classifier(x, leaf):
    """
    Classify an input based on a leaf condition.

    Parameters:
    - x (numpy.ndarray): The input to classify.
    - leaf (list): Leaf condition containing attribute, value, and condition type.

    Returns:
    - label: The classification label if conditions are met, otherwise None.
    """
    attributes, label = leaf[:-1], leaf[-1]
    
    for attribute in attributes:
        attr_index, attr_value, condition_type = attribute
        
        # Handling categorical attributes
        if condition_type == 0:
            if x[attr_index] != attr_value:
                return None
        # Handling numerical attributes
        elif condition_type == 1:
            if x[attr_index] == attr_value:
                return None
    
    return label

class credalDecisionTree():
    """
    Credal Decision Tree implementation.

    Parameters:
    - s (int): Some parameter.
    - threshold (float): Threshold for stopping tree growth.
    """
    
    def __init__(self, s: int, threshold: float = 0.001) -> None:
        """
        Initialize the credal decision tree.

        Parameters:
        - s (int): Some parameter.
        - threshold (float): Threshold for stopping tree growth.
        """
        self.s = s
        self.threshold = threshold
        self.leaf = []

    def fit(self, X: np.ndarray, y: np.ndarray, partitions: list = None) -> None:
        """
        Fit the credal decision tree to the data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.
        - partitions (list): List to store the tree structure.
        """
        if partitions is None:
            partitions = []  

        attribute, threshold, score = find_best_attribute_split(X, y)

        if score < self.threshold:
            label = most_common_value(y)
            partitions.append(label)
            self.leaf.append(partitions)
        else:
            partitions_left = partitions.copy()
            partitions_right = partitions.copy()

            partitions_left.append((attribute, threshold, 0))
            partitions_right.append((attribute, threshold, 1))

            X_left, y_left, X_right, y_right = partition(X, y, attribute, threshold)
            
            if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                label = most_common_value(y)
                partitions.append(label)
                self.leaf.append(partitions)
            else:
                self.fit(X_left, y_left, partitions_left)
                self.fit(X_right, y_right, partitions_right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for a set of input data.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - predictions (numpy.ndarray): Predicted labels.
        """
        predictions = []
        for x in X:
            for leaf in self.leaf:
                out = leaf_classifier(x, leaf)
                if out is not None:
                    predictions.append(out)
                    break
                else:
                    continue
        return np.array(predictions)
