import numpy as np


def add_intercept(x):
    return np.hstack((np.ones((len(x), 1)), x))


def predict_(x, theta):
    X = add_intercept(x)
    return X.dot(theta)


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible shapes.
    Args:
    x: has to be an numpy.array, a matrix of shape m * n.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of shapes n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    X = add_intercept(x)
    m = len(x)
    return (1 / m) * X.T.dot(predict_(x, theta) - y)


x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19])
theta1 = np.array([0, 3, 0.5, -6])
# Example 0:
ret = gradient(x, y, theta1)
print(ret)
# Output:
# array([ -33.71428571, -37.35714286, 183.14285714, -393.])
# Example 1:
theta2 = np.array([0, 0, 0, 0])
print(gradient(x, y, theta2))
# Output:
# array([ -0.71428571, 0.85714286, 23.28571429, -26.42857143])