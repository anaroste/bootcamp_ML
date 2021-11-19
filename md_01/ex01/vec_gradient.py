import numpy as np


def add_intercept(x):
    if x.shape != (len(x), 1):
        x = np.reshape(x, (len(x), 1))
    return np.hstack((np.ones((len(x), 1)), x))


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta is an empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m = len(x)
    X = add_intercept(x)
    return (X / m).T.dot(X.dot(theta) - y)

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
# Example 0:
theta1 = np.array([2, 0.7])
print(gradient(x, y, theta1))