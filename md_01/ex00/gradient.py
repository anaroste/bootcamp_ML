from matplotlib.pyplot import axis
import numpy as np


def add_intercept(x):
    if len(x) == 0:
        return None
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x, theta):
    X = add_intercept(np.reshape(x, (len(x), 1)))
    return np.sum(X * theta, axis=1)


def simple_predict(x, theta):
    if not isinstance(theta, np.ndarray) and len(theta) != 2:
        return None
    return [float(theta[0] + theta[1] * xi) for xi in x]


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    nabla = [0, 0]
    m = len(x)
    y_hat = simple_predict(x, theta)
    nabla[0] = (1 / m) * np.sum(y_hat - y)
    nabla[1] = (1 / m) * np.sum((y_hat - y).dot(x))
    return nabla


x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
# Example 0:
theta1 = np.array([2, 0.7])
print(simple_gradient(x, y, theta1))

theta2 = np.array([1, -0.4])
print(simple_gradient(x, y, theta2))
