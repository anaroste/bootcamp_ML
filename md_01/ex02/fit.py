import numpy as np


def add_intercept(x):
    if x.shape != (len(x), 1):
        x = np.reshape(x, (len(x), 1))
    return np.hstack((np.ones((len(x), 1)), x))


def gradient(x, y, theta):
    m = len(x)
    X = add_intercept(x)
    # print(X.dot(theta))
    return (X / m).T.dot(X.dot(theta) - y)


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape 2 * 1.
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    for i in range(0, max_iter):
        theta = theta - alpha * gradient(x, y.T[0], theta)
    return np.array([[theta[0]], [theta[1]]])

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta = np.array([1, 1])
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
print(theta1)