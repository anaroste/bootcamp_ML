import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same shapes.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    m = len(y)
    return (1 / (2 * m)) * (y_hat - y).dot(y_hat - y)


X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
# Example 0:
print(loss_(X, Y))
# Output:
# 2.1428571428571436
# Example 1:
print(loss_(X, X))
# Output:
# 0.0
