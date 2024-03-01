import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize, Bounds
from itertools import product

class CredalJoint():
    """
    Class for the joint credal set.

    ...

    Attributes
    ----------
    s : int
        The parameter in the Imprecise Dirichlet Model.
    jfs : np.ndarray
        The joint feature space of the joint credal set.
    max_iter : int
        The maximum number of iterations for the optimization algorithm.
    base : int
        The base of the logarithm in the entropy function.
    max_entropy_joint : np.ndarray
        The maximum entropy joint probability distribution.
    min_entropy_joint : np.ndarray
        The minimum entropy joint probability distribution.

    Methods
    -------
    joint_feature_space(data:np.ndarray) -> np.ndarray
        Returns the joint feature space of the joint credal set.
    lower_bound(data:np.ndarray) -> np.ndarray
        Returns the lower bound of the joint credal set.
    upper_bound(data:np.ndarray) -> np.ndarray
        Returns the upper bound of the joint credal set.
    marginal(index:int, entropy_max:bool) -> np.ndarray
        Returns the marginal probability distribution for a given variable.
    conditional(index:int, cond_index:int, cond_value:float, entropy_max:bool, return_complement:bool=False) -> np.ndarray
        Returns the conditional probability distribution for a given variable given a condition.
    information_gain(index:int, cond_index:int, value:int, entropy_max:bool) -> np.ndarray
        Returns the information gain for a given variable given a condition.
    interval_information_gain(index:int, cond_index:int, value:int) -> np.array
        Returns the interval information gain for a given variable given a condition.
    all_interval_ig() -> dict
        Returns the interval information gain for all variables given a condition.
    max_entropy_joint(data:np.ndarray) -> np.ndarray
        Returns the maximum entropy joint probability distribution given the lower and upper bounds of the joint probability space.
    min_entropy_joint(data:np.ndarray) -> np.ndarray
        Returns the minimum entropy joint probability distribution given the lower and upper bounds of the joint probability space.
    """
    def __init__(self, data:np.ndarray, s:int, max_iter:int, base:int) -> None:
        self.s = s
        self.jfs = self.joint_feature_space(data)
        self.max_iter = max_iter
        self.base = base
        self.N = data.shape[0]
        self.max_entropy_joint = self.max_entropy_joint(data)
        self.min_entropy_joint = self.min_entropy_joint(data)

    def joint_feature_space(self, data:np.ndarray) -> np.ndarray:
        """
        Returns the joint feature space of the joint credal set.

            Parameters:
                data (np.ndarray): The data.

            Returns:
                joint_feature_space (np.ndarray): The joint feature space of the joint credal set.
        """
        return np.array(list(product(*[np.unique(data[:,i]) for i in range(data.shape[1])])))
    
    def lower_bound(self, data: np.ndarray) -> np.ndarray:
        """
        Returns the lower bound of the joint credal set.

            Parameters:
            data (np.ndarray): The data.

            Returns:
            lower_bound (np.ndarray): The lower bound of the joint credal set.
        """
        counts = np.zeros(self.N)
        for idx, i in enumerate(self.jfs):
            counts[idx] = np.all(i == data, axis=1).sum()
        return counts / (self.N + self.s)

    def upper_bound(self, data: np.ndarray) -> np.ndarray:
        """
        Returns the upper bound of the joint credal set.
            
                Parameters:
                    data (np.ndarray): The data.
    
                Returns:
                    upper_bound (np.ndarray): The upper bound of the joint credal set.
        """
        counts = np.zeros(self.jfs.shape[0])
        for idx, i in enumerate(self.jfs):
            for d in data:
                mask = ~np.isnan(d)
                if np.all(i[mask] == d[mask]):
                    counts[idx] +=1
        return (counts + self.s) / (data.shape[0] + self.s)

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
            values = np.unique(self.jfs[:,index])
            p = np.zeros(values.shape[0])
            total = 0
            for idx, v in enumerate(values):
                mask = v == self.jfs[:,index]
                p_i = np.sum(self.max_entropy_joint[mask])
                total += p_i
                p[idx] = p_i
            return p / total
        
        else:
            values = np.unique(self.jfs[:,index])
            p = np.zeros(values.shape[0])
            total = 0
            for idx, v in enumerate(values):
                mask = v == self.jfs[:,index]
                p_i = np.sum(self.min_entropy_joint[mask])
                total += p_i
                p[idx] = p_i
            return p / total
    
    def conditional(self, index:int, cond_index:int, cond_value:float, entropy_max:bool, return_complement:bool=False) -> np.ndarray:
        """
        Returns the conditional probability distribution for a given variable given a condition.
        
            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                cond_value (float): The value of the condition.
                entropy_max (bool): If True, returns the conditional probability derived from the joint distribution that maximizes the entropy.
                                    If False, returns the conditional probability derived from the joint distribution that minimizes the entropy.
                return_complement (bool): If True, returns the conditional probability distribution of the complement of the condition.

            Returns:
                conditional_prob (np.ndarray): The conditional probability distribution.
        """
        if entropy_max:
            values = np.unique(self.jfs[:,index])
            p = np.zeros(values.shape[0])
            p_complement = np.zeros(values.shape[0])
            total = 0 
            total_complement = 0
            for idx, v in enumerate(values):
                mask = (v == self.jfs[:,index]) & (v == self.jfs[:,cond_index])
                mask_complement = (v == self.jfs[:,index]) & (cond_value != self.jfs[:,cond_index])
                p_i = np.sum(self.max_entropy_joint[mask])
                p_i_complement = np.sum(self.max_entropy_joint[mask_complement])
                total += p_i
                total_complement += p_i_complement
                p[idx] = p_i
                p_complement[idx] = p_i_complement
            if return_complement:
                return p / total, p_complement / total_complement
            else:
                return p / total
        
        else:
            values = np.unique(self.jfs[:,index])
            p = np.zeros(values.shape[0])
            p_complement = np.zeros(values.shape[0])
            total = 0 
            total_complement = 0
            for idx, v in enumerate(values):
                mask = (v == self.jfs[:,index]) & (v == self.jfs[:,cond_index])
                mask_complement = (v == self.jfs[:,index]) & (cond_value != self.jfs[:,cond_index])
                p_i = np.sum(self.min_entropy_joint[mask])
                p_i_complement = np.sum(self.min_entropy_joint[mask_complement])
                total += p_i
                total_complement += p_i_complement
                p[idx] = p_i
                p_complement[idx] = p_i_complement
            if return_complement:
                return p / total, p_complement / total_complement
            else:
                return p / total
        
    def information_gain(self, index:int, cond_index:int, value:int, entropy_max:bool) -> np.ndarray:
        """
        Returns the information gain for a given variable given a condition.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                value (int): The value of the condition.
                entropy_max (bool): If True, returns the information gain derived from the joint distribution that maximizes the entropy.
                                    If False, returns the information gain derived from the joint distribution that minimizes the entropy.

            Returns:
                ig (np.ndarray): The information gain.
        """
        values = np.unique(self.jfs[:,cond_index])
        p_y = self.marginal(index, entropy_max)
        p_x = self.marginal(cond_index, entropy_max)
        p_x_i = p_x[np.where(values == value)[0][0]]
        p_x_complement = p_x.sum() - p_x_i
        p_y_x_i, p_y_x_complement = self.conditional(index, cond_index, value, entropy_max, True)
        ig = entropy(p_y, base=self.base) - (p_x_i * entropy(p_y_x_i, base=self.base) + p_x_complement * entropy(p_y_x_complement, base=self.base))
        return ig
    
    def interval_information_gain(self, index:int, cond_index:int, value:int) -> np.array:
        """
        Returns the interval information gain for a given variable given a condition.

            Parameters:
                index (int): The index of the variable.
                cond_index (int): The index of the condition.
                value (int): The value of the condition.

            Returns:
                iig (np.ndarray): The interval information gain.
        """
        iig = []
        for entropy_max in [True, False]:
            ig = self.information_gain(index, cond_index, value, entropy_max)
            ig = ig if not np.isnan(ig) else 0
            iig.append(ig)
        return sorted(iig)
    
    def all_interval_ig(self) -> dict:
        """
        Returns the interval information gain for all variables given a condition.

            Returns:
                iig (dict): The interval information gain.
        """
        iig = {}
        for feature in range(self.jfs.shape[1] - 1):
            values = np.unique(self.jfs[:,feature])
            for value in values:
                interval = self.interval_information_gain(-1, feature, value)
                iig[(feature, value)] = interval
        return iig

    def max_entropy_joint(self, data:np.ndarray) -> np.ndarray:
        """
        Returns the maximum entropy joint probability distribution given the lower and upper bounds
        of the joint probability space.

            Parameters:
                data (np.ndarray): The data.

            Returns:
                res.x (np.ndarray): The maximum entropy joint probability distribution.
        """
        lower_bound = self.lower_bound(data)
        upper_bound = self.upper_bound(data)

        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
        bounds = Bounds(lower_bound, upper_bound)
        negative_joint_entropy = lambda x: -entropy(x, base=self.base)
        initial_guess = np.zeros(self.jfs.shape[0])
        res = minimize(negative_joint_entropy, initial_guess, bounds=bounds, constraints=constraints, options={'maxiter': self.max_iter})
        return res.x
    
    def min_entropy_joint(self, data:np.ndarray) -> np.ndarray:
        """
        Returns the minimum entropy joint probability distribution given the lower and upper bounds
        of the joint probability space.

            Parameters:
                data (np.ndarray): The data.

            Returns:
                res.x (np.ndarray): The minimum entropy joint probability distribution.
        """
        lower_bound = self.lower_bound(data)
        upper_bound = self.upper_bound(data)

        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
        bounds = Bounds(lower_bound, upper_bound)
        initial_guess = np.zeros(self.jfs.shape[0])
        res = minimize(entropy, initial_guess, bounds=bounds, constraints=constraints, options={'maxiter': self.max_iter})
        return res.x