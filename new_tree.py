from credal_joint import CredalJoint
import tqdm
import numpy as np

class Node():
    """
    Node class for the decision tree. A node can either be a leaf node or a decision node.
    A decision node is defined by a feature and values associated with that feature.
    A leaf node is defined by a class label.

    ...

    Attributes
    ----------
    feature : int
        The feature associated with the node.
    children : dict
        Dictionary containing the children of the node.
    label : int
        The class label if node is a leaf node.
    """

    def __init__(self, credal_joint:CredalJoint, target:int, lower_ig:bool, threshold) -> None:
        self.lower_ig = lower_ig
        self.threshold = threshold
        self.target = target
        self.credal_joint = credal_joint
        self.feature, self.ig = self.best_split( self.iig_all(credal_joint) )
        self.children = []
        self.label = None
        self.create_children(threshold)
    
    def iig_all(self, joint_credal:CredalJoint) -> dict:
        """
        Returns the interval information gain of all the variables in the data.

            Parameters:
                data (np.ndarray): The data.
                target (int): The index of the target column.
                joint_feature_space (np.ndarray): The joint feature space.
                s (int): The Imprecise Dirichlet Model parameter.

            Returns:
                var (dict): The interval information gain of all the variables in the data.
        """
        var_iig = {}
        for var in range(joint_credal.joint_feature_space.shape[1]):
            if var == self.target:
                continue
            else:
                var_iig[str(var)] = joint_credal.interval_ig(self.target, var)
        return var_iig
    
    def best_split_max(self, var_ig_pairs:dict) -> tuple[int, float]:
            """
            Returns the variable with that maximizes the upper estimates for information gain.

                Parameters:
                    var_ig_pairs (dict): The interval information gain of all the variables in the data.

                Returns:
                    best_var, max_ig (tuple): The variable with the maximum upper estimate information gain and the estimated information gain.
            """
            max_ig = -float('inf')	
            best_var = None
            for var in var_ig_pairs.keys():
                if var_ig_pairs[var][1] > max_ig:
                    max_ig = var_ig_pairs[var][1]
                    best_var = var
            return best_var, max_ig
    
    def best_split_min(self, var_ig_pairs:dict) -> tuple[int, float]:
        """
        Returns the variable with that minimizes the lower estimates for information gain.

            Parameters:
                var_ig_pairs (dict): The interval information gain of all the variables in the data.

            Returns:
                best_var, min_ig (tuple): The variable with the minimum lower estimate information gain and the estimated information gain.
        """
        min_ig = -float('inf')	
        best_var = None
        for var in var_ig_pairs.keys():
            if var_ig_pairs[var][0] > min_ig:
                min_ig = var_ig_pairs[var][0]
                best_var = var
        return best_var, min_ig
    
    def best_split(self, var_ig_pairs:dict) -> dict:
        """
        Return the best split for the given data.

        Parameters:
            data (np.ndarray): The data to split.
            target (int): The index of the target column.
            max_entropy (bool): If True, the split is based on the maximum entropy of the joint.
            joint_feature_space (np.ndarray): The joint feature space.
            s (int): The Imprecise Dirichlet Model parameter.
        
        Returns:
            dict: The best split.
        """
        if self.lower_ig:
            best_var, ig = self.best_split_min(var_ig_pairs)
        else:
            best_var, ig = self.best_split_max(var_ig_pairs)
        return int(best_var), ig
    
    def majority_class_max(self) -> int:
        """
        Returns the majority class of the data.

        Parameters:
            data (np.ndarray): The data.
            target (int): The index of the target column.
            joint_feature_space (np.ndarray): The joint feature space.
            s (int): The Imprecise Dirichlet Model parameter.
        
        Returns:
            int: The majority class of the data.
        """
        marginal = self.credal_joint.marginal(self.target, True)
        return np.argmax(marginal)

    def majority_class_min(self) -> int:
        """
        Returns the majority class of the data.

        Parameters:
            data (np.ndarray): The data.
            target (int): The index of the target column.
            joint_feature_space (np.ndarray): The joint feature space.
            s (int): The Imprecise Dirichlet Model parameter.
        
        Returns:
            int: The majority class of the data.
        """
        marginal = self.credal_joint.marginal(self.target, False)
        return np.argmax(marginal)
    
    def majority_class(self) -> int:
        """
        Returns the majority class of the data.

        Parameters:
            data (np.ndarray): The data.
            target (int): The index of the target column.
            joint_feature_space (np.ndarray): The joint feature space.
            s (int): The Imprecise Dirichlet Model parameter.
            max_entropy (bool): If True, the majority class is based on the maximum entropy of the joint.
        
        Returns:
            int: The majority class of the data.
        """
        if self.lower_ig:
            return self.majority_class_max()
        else:
            return self.majority_class_min()
    
    def create_children(self, threshold:float):
        """
        """
        if self.ig < threshold:
            self.label = self.majority_class()
        else:
            values = np.unique(self.credal_joint.joint_feature_space[:,self.feature])
            for v in values:
                credal_joint_child = self.credal_joint.cond_jc(self.feature, v)
                child_node = Node(credal_joint_child, self.target - 1, self.lower_ig, self.threshold)
                child_node.create_children(threshold)
                self.children.append(child_node)

class DecisionTree():
    """
    Class for the decision tree.

    ...

    Attributes
    ----------
    root : Node
        The root node of the decision tree.
    max_entropy : bool
        If True, the split is based on the maximum entropy of the joint.
    
    Methods
    -------
    predict(sample:np.ndarray) -> int
        Predicts the class label of a given sample.
    """

    def __init__(self, lower_ig:bool, s:int, data:np.ndarray, joint_feature_space:np.ndarray, max_iter:int, base:int, threshold) -> None:
        self.lower_ig = lower_ig
        self.threshold = threshold
        self.s = s
        self.max_iter = max_iter
        self.base = base
        self.credal_joint = CredalJoint(data, joint_feature_space, self.s, self.max_iter, self.base)
        self.target = joint_feature_space.shape[1] - 1
        self.root = self.build_tree()
        
    def build_tree(self, root:Node=None) -> None:
        """
        Builds a decision tree from the given data.

        Parameters:
            data (np.ndarray): The data to split.
            target (int): The index of the target column.
            joint_feature_space (np.ndarray): The joint feature space.
            s (int): The Imprecise Dirichlet Model parameter.
            max_entropy (bool): If True, the split is based on the maximum entropy of the joint.
        
        Returns:
            Node: The root node of the decision tree.
        """
        return Node(self.credal_joint, self.target, self.lower_ig, self.threshold)
        
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
            print(node.feature)
            print(sample)
            print(node.children)
            child_node = node.children[sample[node.feature]]
            return self.predict_single(child_node, np.delete(sample, node.feature))
    
    def predict(self, node:Node, samples:np.ndarray) -> np.ndarray:
        """
        Predicts the class labels of given samples.

        Parameters:
            node (Node): The root node of the decision tree.
            samples (np.ndarray): The samples to predict.
        
        Returns:
            np.ndarray: The predicted class labels of the samples.
        """
        output = []
        for sample in samples:
            output.append(self.predict_single(node, sample))
        return np.array(output)

    def get_depth(self, node:Node) -> int:
        """
        Returns the depth of the decision tree.

        Parameters:
            node (Node): The root node of the decision tree.
        
        Returns:
            int: The depth of the decision tree.
        """
        if node.children == []:
            return 1
        else:
            max_depth = 0
            for child in node.children:
                max_depth = max(max_depth, self.get_depth(child))
            return max_depth + 1