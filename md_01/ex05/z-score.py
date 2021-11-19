import numpy as np


def mean(x):
    ret = 0
    for elt in x:
        ret += elt
    return float(ret / len(x))


def std(x):
    m = sum(x) / len(x)
    return np.sqrt(sum((xi - m) ** 2 for xi in x) / len(x))


def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    return [(xi - mean(x)) / std(x) for xi in x]

x = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(x))

y = np.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(y))