import numpy as np


def add_intercept(x):
    if x.shape != (len(x), 1):
        x = np.reshape(x, (len(x), 1))
    return np.hstack((np.ones((len(x), 1)), x))


def gradient(x, y, theta):
    m = len(x)
    X = add_intercept(x)
    return (X / m).T.dot(X.dot(theta) - y)


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.theta = thetas

    def fit_(self, x, y):
        for i in range(0, self.max_iter):
            grad = gradient(x, y, theta)
            theta = theta - self.alpha * grad
        return theta

    def predict_(self, x):
        X = add_intercept(np.reshape(x, (len(x), 1)))
        return np.array([np.sum(X * self.theta, axis=1)]).T

    def loss_elem_(self, y, y_hat):
        if y_hat.shape != y.shape:
            y_hat = np.reshape(y_hat, (len(y_hat), 1))
        return (y_hat - y)**2

    def loss_(self, y, y_hat):
        m = len(y)
        return 1 / (2 * m) * np.sum(self.loss_elem_(y, y_hat))

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
lr1 = MyLinearRegression([2, 0.7])
# print(lr1.predict_(x))
# print(lr1.loss_elem_(lr1.predict_(x),y))
# print(lr1.loss_(lr1.predict_(x),y))
lr2 = MyLinearRegression([1, 1], 5e-8, 1500000)
lr2.fit_(x, y)
lr2.thetas
lr2.predict_(x)
# print(MyLinearRegression.loss_elem_(lr2.predict_(x),y))
# print(MyLinearRegression.loss_(lr2.predict_(x),y))