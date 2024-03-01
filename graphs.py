from copy import deepcopy

class Node():
    def __init__(self, feature=None, value=None, children=None, label=None) -> None:
        self.feature = feature
        self.value = value
        self.left = None
        self.right = None
        self.label = label
