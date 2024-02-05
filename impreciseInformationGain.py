import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

def credal_set(y: np.ndarray, s: int) -> np.ndarray:
    """
    Compute the credal set for a discrete variable y.

    Parameters:
        y (np.ndarray): Array of observed values for the variable y.
        s (int): Parameter used in the computation of lower and upper bounds.

    Returns:
        np.ndarray: Credal set represented by lower and upper bounds.
    """
    unique_values = np.unique(y)
    lower_bounds = np.array([np.sum(y == x)/(y.shape[0] + s) for x in unique_values])
    upper_bounds = np.array([(np.sum(y == x) + s)/(y.shape[0] + s) for x in unique_values])
    return np.array([lower_bounds, upper_bounds])

def conditional_credal_set(y: np.ndarray, x: np.ndarray, x_val: int, s: int, complement: bool = False) -> np.ndarray:
    """
    Compute the conditional credal set for a discrete variable y given x and x_val.

    Parameters:
        y (np.ndarray): Array of observed values for the variable y.
        x (np.ndarray): Array of observed values for the conditioning variable x.
        x_val (int): The value of x for which the conditional credal set is computed.
        s (int): Parameter used in the computation of lower and upper bounds.
        complement (bool): Flag to compute the complement of the conditional set.

    Returns:
        np.ndarray: Conditional credal set represented by lower and upper bounds.
    """
    unique_y = np.unique(y)
    if complement:
        lower_bounds = np.array([np.sum((y == y_val) & (x != x_val))/(np.sum(x != x_val) + s) for y_val in unique_y])
        upper_bounds = np.array([(np.sum((y == y_val) & (x != x_val)) + s)/(np.sum(x != x_val) + s) for y_val in unique_y])
    else:
        lower_bounds = np.array([np.sum((y == y_val) & (x == x_val))/(np.sum(x == x_val) + s) for y_val in unique_y])
        upper_bounds = np.array([(np.sum((y == y_val) & (x == x_val)) + s)/(np.sum(x == x_val) + s) for y_val in unique_y])
    return np.array([lower_bounds, upper_bounds])

def negative_entropy(x: np.ndarray) -> float:
    """
    Compute the negative entropy of a discrete probability distribution.

    Parameters:
        x (np.ndarray): Array representing a probability distribution.

    Returns:
        float: Negative entropy value.
    """
    return -entropy(x)

def conditional_negative_entropy(x: np.ndarray, x_val: int, y: np.ndarray) -> float:
    """
    Compute the negative entropy of the conditional distribution of y given x=x_val.

    Parameters:
        x (np.ndarray): Array representing the values of the conditioning variable x.
        x_val (int): The value of x for which conditional entropy is computed.
        y (np.ndarray): Array representing the values of the target variable y.

    Returns:
        float: Conditional negative entropy value.
    """
    conditional_y = y[x == x_val]
    return negative_entropy(conditional_y)

def h_star(credal_set: np.ndarray) -> float:
    """
    Compute the maximum entropy within a given credal set.

    Parameters:
        credal_set (np.ndarray): Credal set represented by lower and upper bounds.

    Returns:
        float: Maximum entropy value.
    """
    initial_guess = np.array([1/credal_set.shape[1]] * credal_set.shape[1])

    constraint = {'type': 'eq', 'fun': lambda x: sum(x) - 1}

    result = minimize(negative_entropy, 
                      initial_guess, 
                      constraints=constraint, 
                      bounds=list(zip(credal_set[0], credal_set[1])))
    
    max_objective_value = -result.fun

    return max_objective_value

def probability_of_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the probability distribution for the variable x given observed values y.

    Parameters:
        x (np.ndarray): Array representing the values of the variable x.
        y (np.ndarray): Array representing the values of the target variable y.

    Returns:
        np.ndarray: Probability distribution for variable x.
    """
    unique_x = np.unique(x)
    credal_x = credal_set(x, 1)  # Assuming you have a function credal_set defined
    
    def objective_function(distr_x: np.ndarray, unique_x=unique_x, y=y, x=x):
        objective_value = 0
        for idx, x_val in enumerate(unique_x):
            objective_value += distr_x[idx] * conditional_negative_entropy(x, x_val, y)
        return objective_value

    initial_guess = np.array([1/credal_x.shape[1]] * credal_x.shape[1])

    constraint = {'type': 'eq', 'fun': lambda x: sum(x) - 1}

    result = minimize(objective_function,
                      initial_guess,
                      constraints=constraint,
                      bounds=list(zip(credal_x[0], credal_x[1])))

    return result.x

def imprecise_information_gain(y: np.ndarray, x_val: int, x: np.ndarray, s: int) -> float:
    """
    Compute the imprecise information gain based on the entropy measures and credal sets.

    Parameters:
        y (np.ndarray): Array representing the values of the target variable y.
        x_val (int): The value of x for which the information gain is computed.
        x (np.ndarray): Array representing the values of the conditioning variable x.
        s (int): Parameter used in the computation of credal sets.

    Returns:
        float: Imprecise information gain value.
    """
    unique_x = np.unique(x)

    credal_y = credal_set(y, s)
    conditional_credal_y_left = conditional_credal_set(y, x, x_val, s)
    conditional_credal_y_right = conditional_credal_set(y, x, x_val, s, True)
    
    h_star_y = h_star(credal_y)
    h_star_conditional_y_left = h_star(conditional_credal_y_left)
    h_star_conditional_y_right = h_star(conditional_credal_y_right)
    
    distr_x = probability_of_x(x, y)

    p_x = distr_x[unique_x == x_val]
    p_not_x = np.sum(distr_x[unique_x != x_val])

    out = h_star_y - (p_x * h_star_conditional_y_left + p_not_x * h_star_conditional_y_right)

    return out.item()
