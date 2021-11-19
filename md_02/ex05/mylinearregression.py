import numpy as np


def add_intercept(x):
    return np.hstack((np.ones((len(x), 1)), x))


def gradient(x, y, theta):
    X = add_intercept(x)
    m = len(x)
    return (1 / m) * X.T.dot(X.dot(theta) - y)


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def fit_(self, x, y):
        for i in range(0, self.max_iter):
            grad = gradient(x, y, self.theta)
            self.theta = self.theta - self.alpha * grad

    def predict_(self, x):
        X = add_intercept(x)
        return X.dot(self.theta)

    def loss_elem_(self, y, y_hat):
        if y_hat.shape != y.shape:
            y_hat = np.reshape(y_hat, (len(y_hat), 1))
        return (y_hat - y)**2

    def loss_(self, y, y_hat):
        m = len(y)
        return 1 / (2 * m) * np.sum(self.loss_elem_(y, y_hat))


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
# Example 0:
y_hat = mylr.predict_(X)
print(y_hat)
# Output:
# array([[8.], [48.], [323.]])
# Example 1:
print(mylr.loss_elem_(Y,y_hat))
# Output:
# array([[225.], [0.], [11025.]])
# Example 2:
print(mylr.loss_(Y,y_hat))
# Output:
# 1875.0
# Example 3:
print('---------------------')
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)
# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
# Example 4:
y_hat = mylr.predict_(X)
print(y_hat)
# Output:
# array([[23.417..], [47.489..], [218.065...]])
# Example 5:
print(mylr.loss_elem_(Y,y_hat))
# Output:
# array([[0.174..], [0.260..], [0.004..]])
# Example 6:
print(mylr.loss_(Y,y_hat))
# Output:
# 0.0732..