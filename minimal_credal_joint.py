import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize, Bounds
from itertools import product
import numpy as np
from scipy.stats import entropy
import time

class CredalJoint():
    def __init__(self, s:int, maxiter:int, base:int) -> None:
        self.s = s
        self.maxiter = maxiter
        self.base = base
        self.max_entropy_joint = None
        self.min_entropy_joint = None
        self.joint_space = None
    
    def lower_bound(self, data: np.ndarray, joint_space: np.ndarray) -> np.ndarray:
        """
        Returns the lower bound of the joint credal set.

        Parameters:
        data (np.ndarray): The data.
        joint_space (np.ndarray): The joint space.

        Returns:
        lower_bound (np.ndarray): The lower bound of the joint credal set.
        """
        counts = np.sum(np.all(joint_space[:, None, :] == data[None, :, :], axis=2), axis=1)
        return counts / (data.shape[0] + self.s)
    
    def upper_bound(self, data: np.ndarray, joint_space: np.ndarray) -> np.ndarray:
        nan_loc = np.isnan(data)
        eval = joint_space[:, None, :] == data[None, :, :]
        eval[:,nan_loc] = True
        counts = np.sum(np.all(eval, axis=2), axis=1)
        return (counts + self.s) / (data.shape[0] + self.s)

    def optimizer(self, lower_bound:np.ndarray, upper_bound:np.ndarray):
        bounds = Bounds(lower_bound, upper_bound)
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)})
        negative_entropy = lambda x: -entropy(x, base=self.base)
        min_entropy = minimize(entropy, np.zeros(lower_bound.shape[0]), constraints=constraints, bounds=bounds, options={'maxiter':self.maxiter}).x
        max_entropy = minimize(negative_entropy, np.zeros(lower_bound.shape[0]), constraints=constraints, bounds=bounds, options={'maxiter':self.maxiter}).x
        return min_entropy, max_entropy

    def fit(self, x:np.ndarray, y:np.ndarray, threshold:float) -> None:
        x = x > threshold
        x = x.astype(int)
        x_values = np.unique(x[~np.isnan(x)])
        y_values = np.unique(y[~np.isnan(y)])
        joint_space = np.transpose([np.tile(x_values, len(y_values)), np.repeat(y_values, len(x_values))])
        lb = self.lower_bound(np.column_stack((x, y)), joint_space)
        ub = self.upper_bound(np.column_stack((x, y)), joint_space)
        min_entropy, max_entropy = self.optimizer(lb, ub)
        self.max_entropy_joint = max_entropy
        self.min_entropy_joint = min_entropy
        self.joint_space = joint_space

    def marginal(self, target: bool = True, complement: bool = False):
        index = 1 if target else 0
        values = np.unique(self.joint_space[:, index])
        p_y_max = np.zeros(len(np.unique(self.joint_space[:, index])))
        p_y_min = np.zeros(len(np.unique(self.joint_space[:, index])))
        p_y_max_comp = np.zeros(len(np.unique(self.joint_space[:, index])))
        p_y_min_comp = np.zeros(len(np.unique(self.joint_space[:, index])))
        for idx, i in enumerate(values):
            mask = self.joint_space[:, index] == i
            p_y_max[idx] = np.sum(self.max_entropy_joint[mask])
            p_y_min[idx] = np.sum(self.min_entropy_joint[mask])
            p_y_max_comp[idx] = np.sum(self.max_entropy_joint[~mask])
            p_y_min_comp[idx] = np.sum(self.min_entropy_joint[~mask])
        if complement:
            return values, p_y_max, p_y_min, p_y_max_comp, p_y_min_comp
        else:
            return values, p_y_max, p_y_min
    
    def joint(self, x_val: int, y_val: int):
        mask = (self.joint_space[:, 0] == x_val) & (self.joint_space[:, 1] == y_val)
        return self.max_entropy_joint[mask], self.min_entropy_joint[mask]

    def conditional(self, value: int, complement: bool = False):
        comp_value = np.abs(value - 1)
        y_values = np.unique(self.joint_space[:, 1])
        p_y_given_x_max = np.zeros(len(y_values))
        p_y_given_x_min = np.zeros(len(y_values))
        p_y_given_x_max_comp = np.zeros(len(y_values))
        p_y_given_x_min_comp = np.zeros(len(y_values))
        x_values, p_x_max, p_x_min, p_x_max_comp, p_x_min_comp = self.marginal(False, True)
        for idx, y_val in enumerate(y_values):
            p_y_given_x_max[idx], p_y_given_x_min[idx] = self.joint(value, y_val)
            p_y_given_x_max[idx] /= p_x_max[np.where(x_values == value)[0]]
            p_y_given_x_min[idx] /= p_x_min[np.where(x_values == value)[0]]
            if comp_value in x_values:
                p_y_given_x_max_comp[idx], p_y_given_x_min_comp[idx] = self.joint(comp_value, y_val)
                p_y_given_x_max_comp[idx] /= p_x_max_comp[np.where(x_values == comp_value)[0]]
                p_y_given_x_min_comp[idx] /= p_x_min_comp[np.where(x_values == comp_value)[0]]
            else:
                p_y_given_x_max_comp[idx] = 0
                p_y_given_x_min_comp[idx] = 0
        if complement:
            return p_y_given_x_max, p_y_given_x_min, p_y_given_x_max_comp, p_y_given_x_min_comp
        return p_y_given_x_max, p_y_given_x_min
    
    def information_gain(self, value:int):
        _, p_y_max, p_y_min = self.marginal()
        p_y = np.vstack((p_y_max, p_y_min))

        values_x, p_x_max, p_x_min = self.marginal(False)
        p_x_max = p_x_max[np.where(values_x == value)[0]]
        p_x_min = p_x_min[np.where(values_x == value)[0]]
        p_x = np.array([p_x_max, p_x_min])

        p_x_comp = 1 - p_x

        p_y_x_max, p_y_x_min, p_y_x_max_comp, p_y_min_comp = self.conditional(value, True)
        p_y_x = np.vstack((p_y_x_max, p_y_x_min))
        p_y_x_comp = np.vstack((p_y_x_max_comp, p_y_min_comp))

        ig = -1 * p_y * np.emath.logn(self.base, p_y)
        ig[np.isnan(ig)] = 0
        ig = np.sum(ig, axis=1)

        tmp1 = p_x * -1 * p_y_x * np.emath.logn(self.base, p_y_x)
        tmp1[np.isnan(tmp1)] = 0
        tmp1 = np.sum(tmp1, axis=1)

        tmp2 = p_x_comp * -1 *  p_y_x_comp * np.emath.logn(self.base, p_y_x_comp)
        tmp2[np.isnan(tmp2)] = 0
        tmp2 = np.sum(tmp2, axis=1)

        ig -= (tmp1 + tmp2)

        return ig
    