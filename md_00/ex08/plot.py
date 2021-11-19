from matplotlib import pyplot as plt
import numpy as np


def add_intercept(x):
    if len(x) == 0:
        return None
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x, theta):
    X = add_intercept(np.reshape(x, (len(x), 1)))
    return np.sum(X * theta, axis=1)


def loss_elem_(y, y_hat):
    if y_hat.shape != y.shape:
        y_hat = np.reshape(y_hat, (len(y_hat), 1))
    return (y_hat - y)**2


def loss_(y, y_hat):
    m = len(y)
    return 1 / (2 * m) * np.sum(loss_elem_(y, y_hat))


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    line = np.linspace(1, 5, 2)
    y_hat = predict_(x, theta)
    plt.plot(x, y, 'o')
    plt.plot(line, theta[0] + theta[1] * line)
    plt.title(f"Cost = {round(loss_(y, y_hat) * 2, 6)}")
    for i in range(len(x)):
        if y[i] > y_hat[i]:
            min = y_hat[i]
            max = y[i]
        else:
            max = y_hat[i]
            min = y[i]
        plt.vlines(x[i], ymin=min, ymax=max, color='red', linestyle='dashed')
    plt.show()


x = np.arange(1, 6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1 = np.array([18, -1])
plot_with_loss(x, y, theta1)

theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)
