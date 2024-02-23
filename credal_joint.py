import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize, Bounds

class CredalJoint():
    """
    Class for the joint credal set. This class is used to compute the marginal and conditional
    which maximizes or minimizes the entropy of the joint probability distribution. 
    
    ...
    
    Attributes
    ----------
    data : np.ndarray
        The data.
    joint_feature_space : np.ndarray
        The joint feature space of the joint credal set.
    lower_bound : np.ndarray
        The lower bound of the joint credal set.
    upper_bound : np.ndarray
        The upper bound of the joint credal set.
    max_entropy_joint : np.ndarray
        The maximum entropy joint probability distribution.
    min_entropy_joint : np.ndarray
        The minimum entropy joint probability distribution.
    s : int
        The parameter in the Imprecise Dirichlet Model.
    
    Methods
    -------
    marginal(index:int, entropy_max:bool) -> np.ndarray
        Returns the marginal probability distribution for a given variable.
    conditional(index:int, cond_index:int, cond_value:float, entropy_max:bool) -> np.ndarray
        Returns the conditional probability distribution for a given variable given a condition.
    interval_ig(index:int, cond_index:int, entropy_max:bool) -> float
        Returns the interval information gain for a given variable given a condition.

    """
    def __init__(self, data:np.ndarray, joint_feature_space:np.ndarray, s:int, max_iter:int, base:int, max_entropy_joint=None, min_entropy_joint=None) -> None:
        self.data = data
        self.joint_feature_space = joint_feature_space
        self.s = s
        self.max_iter = max_iter
        self.base = base
        self.lower_bound = self.lower_bound(data, joint_feature_space, s)
        self.upper_bound = self.upper_bound(data, joint_feature_space, s)
        self.max_entropy_joint = self.max_entropy_joint(joint_feature_space, self.lower_bound, self.upper_bound) if max_entropy_joint is None else max_entropy_joint
        self.min_entropy_joint = self.min_entropy_joint(joint_feature_space, self.lower_bound, self.upper_bound) if min_entropy_joint is None else min_entropy_joint

    def lower_bound(self, data: np.ndarray, joint_feature_space: np.ndarray, s: int) -> np.ndarray:
            """
            Returns the lower bound of the joint credal set.

                Parameters:
                    data (np.ndarray): The data.
                    joint_feature_space (np.ndarray): The joint feature space of the joint credal set.
                    s (int): represents the parameter in the Imprecise Dirichlet Model.

                Returns:    
                    lower_bound (np.ndarray): The lower bound of the joint credal set.
            """
            counts = np.sum(np.all(data[:, None, :] == joint_feature_space[None, :, :], axis=2), axis=0)
            lower_bound = counts / (data.shape[0] + s)
            return lower_bound

    def upper_bound(self, data: np.ndarray, joint_feature_space:np.ndarray, s:int) -> np.ndarray:
        """
        Returns the upper bound of the joint credal set.

            Parameters:
                data (np.ndarray): The data.
                joint_feature_space (np.ndarray): The joint feature space of the joint credal set.
                s (int): represents the parameter in the Imprecise Dirichlet Model.

            Returns:
                upper_bound (np.ndarray): The upper bound of the joint credal set.
        """
        mask = ~np.isnan(data)
        masked_data = data[mask].reshape(data.shape[0], -1)

        # Expand dimensions for broadcasting
        joint_feature_space_expanded = joint_feature_space[:, None, :]
        masked_data_expanded = masked_data[None, :, :]

        # Compare joint feature space with masked data to find matches
        matches = np.all(joint_feature_space_expanded == masked_data_expanded, axis=2)

        # Count matches and calculate upper bound 
        counts = np.sum(matches, axis=1)
        upper_bound = (counts + s) / (masked_data.shape[0] + s)

        return upper_bound
    
    def joint_entropy(self, joint_prob: np.ndarray):
        return entropy(joint_prob, base=self.base)

    def negative_joint_entropy(self, joint_prob: np.ndarray):
        return -self.joint_entropy(joint_prob)

    def marginal(self, index:int, entropy_max:bool) -> np.ndarray:
        """
        Returns the marginal probability distribution for a given variable.

            Parameters:
                index (int): The index of the variable.
                entropy_max (bool): If True, returns the marginal probability derived from the joint distribution that maximizes the entropy.
                                    If False, returns the marginal probability derived from the joint distribution that minimizes the entropy.
            
            Returns:
                marginal_prob (np.ndarray): The marginal probability distribution.
        """
        if entropy_max:
            marginal_prob = np.bincount(self.joint_feature_space[:, index], weights=self.max_entropy_joint)
        else:
            marginal_prob = np.bincount(self.joint_feature_space[:, index], weights=self.min_entropy_joint)
        return marginal_prob
    
    def conditional(self, index:int, cond_index:int, cond_value:float, entropy_max:bool) -> np.ndarray:
        """
        Returns the conditional probability distribution for a given variable given a condition.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                cond_value (float): The value of the condition.
                entropy_max (bool): If True, returns the conditional probability distribution that maximizes the entropy.
                                    If False, returns the conditional probability distribution that minimizes the entropy.
            
            Returns:
                conditional_prob (np.ndarray): The conditional probability distribution.
        """
        unique_values = np.unique(self.joint_feature_space[:,index])
        conditional_prob = np.zeros(len(unique_values))
        if entropy_max:
            for i, value in enumerate(unique_values):
                mask = (self.joint_feature_space[:,index] == value) & (self.joint_feature_space[:,cond_index] == cond_value)
                conditional_prob[i] = np.sum(self.max_entropy_joint[mask]) / np.sum(self.max_entropy_joint[self.joint_feature_space[:,cond_index] == cond_value])
        else:
            for i, value in enumerate(unique_values):
                mask = (self.joint_feature_space[:,index] == value) & (self.joint_feature_space[:,cond_index] == cond_value)
                conditional_prob[i] = np.sum(self.min_entropy_joint[mask]) / np.sum(self.min_entropy_joint[self.joint_feature_space[:,cond_index] == cond_value])
        return conditional_prob

    def interval_ig(self, index:int, cond_index:int) -> float:
        """
        Returns the interval information gain for a given variable given a condition.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
            
            Returns:
                iig (list): The interval information gain.
        """
        iig = []
        for entropy_max in [True, False]:
            marginal_prob_class = self.marginal(index, entropy_max)
            marginal_prob_cond = self.marginal(cond_index, entropy_max)
            rhs = []
            for value in np.unique(self.joint_feature_space[:,cond_index]):
                conditional_prob = self.conditional(index, cond_index, value, entropy_max)
                rhs.append(np.sum(conditional_prob * np.emath.logn(n=self.base, x=conditional_prob)))
            rhs = np.array(rhs)
            rhs = np.sum(rhs * marginal_prob_cond)
            ig = self.joint_entropy(marginal_prob_class) - rhs
            ig = 0 if np.isnan(ig) else ig
            iig.append(ig)
        return sorted(iig)
    
    def max_entropy_joint(self, joint_feature_space:np.ndarray, lower_bound:np.ndarray, upper_bound:np.ndarray) -> np.ndarray:
        """
        Returns the maximum entropy joint probability distribution given the lower and upper bounds
        of the joint probability space.

            Parameters:
                joint_feature_space (np.ndarray): The joint feature space of the joint credal set.
                lower_bound (np.ndarray): The lower bound of the joint credal set.
                upper_bound (np.ndarray): The upper bound of the joint credal set.

            Returns:
                res.x (np.ndarray): The maximum entropy joint probability distribution.

        """
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
        bounds = Bounds(lower_bound, upper_bound)
        initial_guess = np.repeat(1/joint_feature_space.shape[0], joint_feature_space.shape[0])
        res = minimize(self.negative_joint_entropy, initial_guess, bounds=bounds, constraints=constraints, options={'maxiter': self.max_iter})
        return res.x
    
    def min_entropy_joint(self, joint_feature_space:np.ndarray, lower_bound:np.ndarray, upper_bound:np.ndarray) -> np.ndarray:
        """
        Returns the minimum entropy joint probability distribution given the lower and upper bounds
        of the joint probability space.

            Parameters:
                joint_feature_space (np.ndarray): The joint feature space of the joint credal set.
                lower_bound (np.ndarray): The lower bound of the joint credal set.
                upper_bound (np.ndarray): The upper bound of the joint credal set.

            Returns:
                res.x (np.ndarray): The minimum entropy joint probability distribution.
        """
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
        bounds = Bounds(lower_bound, upper_bound)
        initial_guess = np.repeat(1/joint_feature_space.shape[0], joint_feature_space.shape[0])
        res = minimize(self.joint_entropy, initial_guess, bounds=bounds, constraints=constraints, options={'maxiter': self.max_iter})
        return res.x
    
    def mar_joint_max(self, index:int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint probability whereby a variable is marginalized.

            Parameters:
                index (int): The index of the variable.

            Returns:
                mar_joint (np.ndarray): The joint probability whereby a variable is marginalized.
        """
        reduced_joint_feature_space = np.concatenate((self.joint_feature_space[:,:index], self.joint_feature_space[:,index+1:]), axis=1)
        marginalized_joint_feature_space, inverse_indices = np.unique(reduced_joint_feature_space, axis=0, return_inverse=True)
        marginalized_joint = np.bincount(inverse_indices, weights=self.max_entropy_joint)
        return marginalized_joint_feature_space, marginalized_joint
    
    def mar_joint_min(self, index:int) -> np.ndarray:
        """
        Returns the joint probability whereby a variable is marginalized.

            Parameters:
                index (int): The index of the variable.

            Returns:
                mar_joint (np.ndarray): The joint probability whereby a variable is marginalized.
        """
        reduced_joint_feature_space = np.concatenate((self.joint_feature_space[:,:index], self.joint_feature_space[:,index+1:]), axis=1)
        marginalized_joint_feature_space, inverse_indices = np.unique(reduced_joint_feature_space, axis=0, return_inverse=True)
        marginalized_joint = np.bincount(inverse_indices, weights=self.min_entropy_joint)
        return marginalized_joint_feature_space, marginalized_joint

    def cond_joint_max(self, cond_index:int, cond_value:float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint probability whereby a variable is conditioned.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                cond_value (float): The value of the condition.

            Returns:
                cond_joint (np.ndarray): The joint probability whereby a variable is conditioned.
        """
        mask = self.joint_feature_space[:,cond_index] == cond_value
        reduced_joint_feature_space = self.joint_feature_space[mask]
        reduced_joint_feature_space = np.concatenate((reduced_joint_feature_space[:,:cond_index], reduced_joint_feature_space[:,cond_index+1:]), axis=1)
        conditioned_joint_feature_space, inverse_indices = np.unique(reduced_joint_feature_space, axis=0, return_inverse=True)
        conditioned_joint = np.bincount(inverse_indices, weights=self.max_entropy_joint[mask])
        conditioned_joint /= np.sum(conditioned_joint)
        return conditioned_joint_feature_space, conditioned_joint
    
    def cond_joint_min(self, cond_index:int, cond_value:float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the joint probability whereby a variable is conditioned.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                cond_value (float): The value of the condition.

            Returns:
                cond_joint (np.ndarray): The joint probability whereby a variable is conditioned.
        """
        mask = self.joint_feature_space[:,cond_index] == cond_value
        reduced_joint_feature_space = self.joint_feature_space[mask]
        reduced_joint_feature_space = np.concatenate((reduced_joint_feature_space[:,:cond_index], reduced_joint_feature_space[:,cond_index+1:]), axis=1)
        conditioned_joint_feature_space, inverse_indices = np.unique(reduced_joint_feature_space, axis=0, return_inverse=True)
        conditioned_joint = np.bincount(inverse_indices, weights=self.min_entropy_joint[mask])
        conditioned_joint /= np.sum(conditioned_joint)
        return conditioned_joint_feature_space, conditioned_joint
    
    def cond_jc(self, cond_index:int, cond_value:float):
        """
        Returns the joint credal set whereby a variable is conditioned.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                cond_value (float): The value of the condition.

            Returns:
                cond_jc (CredalJoint): The joint credal set whereby a variable is conditioned.
        """
        cond_data = self.data[self.data[:,cond_index] == cond_value]
        cond_data = np.delete(cond_data, cond_index, axis=1)
        cond_joint_feature_space = self.joint_feature_space[self.joint_feature_space[:,cond_index] == cond_value]
        cond_joint_feature_space = np.delete(cond_joint_feature_space, cond_index, axis=1)
        _, max_joint = self.cond_joint_max(cond_index, cond_value)
        _, min_joint = self.cond_joint_min(cond_index, cond_value)
        return CredalJoint(cond_data, cond_joint_feature_space, self.s, self.max_iter, self.base, max_joint, min_joint)