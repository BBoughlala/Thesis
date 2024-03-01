from credal_joint import CredalJoint
import numpy as np
from graphs import Node
import numpy as np

class DecisionTree():
    """
    A class to represent a decision tree capable of handling imprecise data and missing values.
    The decision tree is built using the Imprecise Dirichlet Model and credal sets.

    ...

    Attributes:
        s : int 
            The Imprecise Dirichlet Model parameter.
        data : np.ndarray: 
            The data.
        max_iter : int: 
            The maximum number of iterations for the optimization procedure to estimate the joints.
        base : int 
            The base utilized in the entropy function.
        root : Node: 
            The root node of the decision tree.
    
    Methods:
        best_split(credal_joint:CredalJoint) -> np.ndarray
            Returns the best split based on the credal joint.
        partition(data:np.ndarray, feature:int, value:int) -> np.ndarray
            Partitions the data based on a feature and value.
        majority_class(data:np.ndarray) -> int
            Returns the majority class of the data.
        build_tree(data:np.ndarray, root:Node=None, credal_joint:np.ndarray=None) -> Node 
            Builds the decision tree.
        predict_single(node:Node, sample:np.ndarray) -> int
            Predicts the class label of a given sample.
        predict(node:Node, samples:np.ndarray) -> np.ndarray
            Predicts the class labels of given samples.
        get_depth(node:Node=None) -> int
            Returns the depth of the decision tree.
    """
    def __init__(self, s:int, max_iter:int, base:int, index_continous:list, n_cat:int) -> None:
        self.s = s
        self.max_iter = max_iter
        self.base = base
        self.index_continous = index_continous
        self.n_cat = n_cat
        self.root = None
     
    def best_split(self, credal_joint:CredalJoint) -> np.ndarray:
        """
        Returns the best split based on the credal joint.

            Parameters:
                credal_joint (CredalJoint): The credal joint.

            Returns:
                np.ndarray: The best split based on the credal joint.
        """
        iig = credal_joint.all_interval_ig()
        best_lower_bound = max(iig.values(), key=lambda x: x[0])
        best_features = [key for key, value in iig.items() if value >= best_lower_bound]
        return best_features
    
    def partition(self, data:np.ndarray, feature:int, value:int) -> np.ndarray:
        """
        Partitions the data based on a feature and value.

            Parameters:
                data (np.ndarray): The data.
                feature (int): The index of the feature.
                value (int): The value of the feature.

            Returns:
                np.ndarray: The partitioned data.
        """
        return data[data[:,feature] == value], data[data[:,feature] != value]
    
    def majority_class(self, data:np.ndarray) -> int:
        """
        Returns the majority class of the data.

            Parameters:
                data (np.ndarray): The data.

            Returns:
                int: The majority class of the data.
        """
        counts = np.bincount(data.astype(int))
        return np.argmax(counts)
    
    def build_tree(self, data:np.ndarray, root:Node=None, credal_joint:np.ndarray=None) -> Node:
        """
        Builds the decision tree.

            Parameters:
                data (np.ndarray): The data.
                root (Node): The root node of the decision tree.
                credal_joint (np.ndarray): The credal joint.

            Returns:
                Node: The root node of the decision tree.
        """
        root = Node() if root is None else root
        credal_joint = CredalJoint(data, self.s, self.max_iter, self.base) if credal_joint is None else credal_joint
        if len(np.unique(data[:,-1])) == 1:
            root.label = data[0,-1]
            return root
        else:
            best_features = self.best_split(credal_joint)
            if len(best_features) > 1:
                best_features = np.random.choice(best_features, 1)
                root.feature = best_features[0]
                root.value = best_features[1]
                left_data, right_data = self.partition(data, root.feature, root.value)
                if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                    root.label = self.majority_class(data[:,-1])
                    return root
                else:
                    left_child = Node()
                    right_child = Node()
                    root.left = self.build_tree(left_data, left_child, credal_joint)
                    root.right = self.build_tree(right_data, right_child, credal_joint)
            else:
                best_features = best_features[0]
                root.feature = best_features[0]
                root.value = best_features[1]
                left_data, right_data = self.partition(data, root.feature, root.value)
                if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                    root.label = self.majority_class(data[:,-1])
                    return root
                else:
                    left_child = Node()
                    right_child = Node()
                    root.left = self.build_tree(left_data, left_child, credal_joint)
                    root.right = self.build_tree(right_data, right_child, credal_joint)
        return root
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Fits the decision tree to the data.

        Parameters:
            X (np.ndarray): The data.
            y (np.ndarray): The class labels.
        """
        y = y.reshape(-1,1)
        data = np.concatenate((X,y), axis=1)
        self.root = self.build_tree(data)
                
    def predict_single(self, node:Node, sample:np.ndarray) -> int:
        """
        Predicts the class label of a given sample.

        Parameters:
            node (Node): The root node of the decision tree.
            sample (np.ndarray): The sample to predict.
        
        Returns:
            int: The predicted class label of the sample.
        """
        if node.label is not None:
            return node.label
        else:
            index = node.feature
            value = node.value
            if sample[index] == value:
                return self.predict_single(node.left, sample)
            else:
                return self.predict_single(node.right, sample)
    
    def predict(self, samples:np.ndarray, node:Node=None) -> np.ndarray:
        """
        Predicts the class labels of given samples.

        Parameters:
            node (Node): The root node of the decision tree.
            samples (np.ndarray): The samples to predict.
        
        Returns:
            np.ndarray: The predicted class labels of the samples.
        """
        node = self.root if node is None else node
        output = []
        for sample in samples:
            output.append(self.predict_single(node, sample))
        return np.array(output)

    def get_depth(self, node:Node=None) -> int:
        """
        Returns the depth of the decision tree.

        Parameters:
            node (Node): The root node of the decision tree.
        
        Returns:
            int: The depth of the decision tree.
        """
        if node is None:
            node = self.root
        if node.left is None and node.right is None:
            return 1
        else:
            return 1 + max(self.get_depth(node.left), self.get_depth(node.right))
    

            