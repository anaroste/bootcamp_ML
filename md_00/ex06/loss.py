import numpy as np


def add_intercept(x):
    if len(x) == 0:
        return None
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x, theta):
    if isinstance(x[0], np.ndarray):
        X = add_intercept(x)
    else:
        X = add_intercept(np.reshape(x, (len(x), 1)))
    return np.sum(X.dot(theta), axis=1)


def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between y and y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if y_hat.shape != y.shape:
        y_hat = np.reshape(y_hat, (len(y_hat), 1))
    return (y_hat - y)**2


def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a shape matching problem between y or y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m = len(y)
    return 1 / (2 * m) * np.sum(loss_elem_(y, y_hat))


x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# Example 1:
print(loss_elem_(y1, y_hat1))
# Output:
# array([[0.], [1], [4], [9], [16]])
# Example 2:
print(loss_(y1, y_hat1))
# Output:
# 3.0
x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])
# Example 3:
print(loss_elem_(y2, y_hat2))
# Output:
# array([[10.5625], [6.0025], [0.1225], [17.2225]])
# Example 4:
print(loss_(y2, y_hat2))
# Output:
# 4.238750000000004
x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])
# Example 5:
print(loss_(y3, y_hat3))
# Output:
# 2.142857142857143
# Example 6:
print(loss_(y3, y3))
# Output
# 0.0