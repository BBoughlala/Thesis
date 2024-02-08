import numpy as np

def is_possible_completion(x: np.ndarray, y: np.ndarray):
    # Ensure y contains only numeric values
    y_numeric = np.asarray(y, dtype=float)
    
    # Create mask for non-NaN values in y
    mask = np.logical_not(np.isnan(y_numeric))
    
    # Perform element-wise comparison between non-NaN elements of x and y
    comparison = x[mask] == y_numeric[mask]
    
    # Check if all comparisons are True
    out = np.all(comparison)
    
    return out

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def get_joint_interval(x: np.ndarray, values: tuple):
    possible_values = cartesian_product(values)
    for possible_v in possible_values:
        lower_count = np.sum(np.all(x == possible_v, axis=1))
        lower_prob = lower_count / x.shape[0]
        upper_count = lower_count
        unknown = x[np.sum(np.isnan(x.astype(float)), axis=1) > 0]
        for i in unknown:
            if is_possible_completion(possible_v, i):
                upper_count += 1
        upper_prob = upper_count / x.shape[0]
        print(f'{possible_v} has a probability between {lower_prob} and {upper_prob}')