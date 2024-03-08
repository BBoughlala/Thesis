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
        x_values = np.unique(x)
        y_values = np.unique(y)
        joint_space = np.transpose([np.tile(x_values, len(y_values)), np.repeat(y_values, len(x_values))])
        lb = self.lower_bound(np.column_stack((x, y)), joint_space)
        ub = self.upper_bound(np.column_stack((x, y)), joint_space)
        min_entropy, max_entropy = self.optimizer(lb, ub)
        self.max_entropy_joint = max_entropy
        self.min_entropy_joint = min_entropy
        self.joint_space = joint_space

    def marginal(self, target: bool = True):
        index = 1 if target else 0
        values = np.unique(self.joint_space[:, index])
        p_y_max = np.zeros(len(np.unique(self.joint_space[:, index])))
        p_y_min = np.zeros(len(np.unique(self.joint_space[:, index])))
        for idx, i in enumerate(values):
            mask = self.joint_space[:, index] == i
            p_y_max[idx] = np.sum(self.max_entropy_joint[mask])
            p_y_min[idx] = np.sum(self.min_entropy_joint[mask])
        return values, p_y_max, p_y_min

    def conditional(self, value: int, complement: bool = False):
        mask = self.joint_space[:, 0] == value
        comp_mask = self.joint_space[:, 0] != value
        p_y_x_max = np.zeros(len(np.unique(self.joint_space[:, 1])))
        p_y_x_min = np.zeros(len(np.unique(self.joint_space[:, 1])))
        p_y_x_max_comp = np.ones(len(np.unique(self.joint_space[:, 1])))
        p_y_x_min_comp = np.ones(len(np.unique(self.joint_space[:, 1])))
        for idx, i in enumerate(np.unique(self.joint_space[:, 1])):
            mask_y = self.joint_space[:, 1] == i
            p_y_x_max[idx] = np.sum(self.max_entropy_joint[mask & mask_y])
            p_y_x_min[idx] = np.sum(self.min_entropy_joint[mask & mask_y])
            p_y_x_max_comp[idx] = np.sum(self.max_entropy_joint[comp_mask & mask_y])
            p_y_x_min_comp[idx] = np.sum(self.min_entropy_joint[comp_mask & mask_y])
        p_y_x_max = p_y_x_max / self.marginal(False)[1][np.where(self.marginal(False)[0] == value)[0]]
        p_y_x_min = p_y_x_min / self.marginal(False)[2][np.where(self.marginal(False)[0] == value)[0]]
        p_y_x_max_comp = p_y_x_max_comp / (1 - self.marginal(False)[1][np.where(self.marginal(False)[0] == value)[0]])
        p_y_x_min_comp = p_y_x_min_comp / (1 - self.marginal(False)[2][np.where(self.marginal(False)[0] == value)[0]])
        if complement:
            return p_y_x_max, p_y_x_min, p_y_x_max_comp, p_y_x_min_comp
        return p_y_x_max, p_y_x_min
    
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
    