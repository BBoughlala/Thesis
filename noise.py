import numpy as np

def add_noise_categorical(x:np.ndarray, p:float, index:int) -> np.ndarray:
    """
    Add noise to the input signal x
    Args:
        x: input signal
        p: noise level
        index: index of the variable on which the noise is added
    Returns:
        x_noise: dataset containing noisy signal
    """
    x_noise = x.copy()
    x_values = np.unique(x[~np.isnan(x[:,index]), index])
    mask = np.random.choice([True, False], size=x_noise.shape[0], p=[p, 1-p])
    x_noise[mask, index] = np.random.choice(x_values, size=np.sum(mask))
    return x_noise

def add_cmar(x:np.ndarray, p:float) -> np.ndarray:
    """
    Add missing values to the input signal x
    Args:
        x: input signal
        p: missing value level
        index: index of the variable on which the missing values are added
    Returns:
        x_missing: dataset containing missing values
    """
    x_missing = x.copy().astype(np.float16)
    mask = np.random.choice([True, False], size=x_missing.shape[0], p=[p, 1-p])
    x_missing[mask] = np.nan
    return x_missing

def add_mnar(x:np.ndarray, p:float) -> np.ndarray:
    """
    Add missing values to the input signal x
    Args:
        x: input signal
        p: missing value level
        index: index of the variable on which the missing values are added
    Returns:
        x_missing: dataset containing missing values
    """
    threshold = np.random.choice(x, 1)
    x_missing = x.copy().astype(np.float16)
    mask = x_missing > threshold
    nan_mask = np.random.choice([True, False], size=x_missing.shape[0], p=[p, 1-p])
    mask = mask & nan_mask
    x_missing[mask] = np.nan
    return x_missing