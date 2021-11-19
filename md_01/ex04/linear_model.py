import pandas as pd
from matplotlib import pyplot as plt
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
        self.loss_history = np.zeros(self.max_iter)
        for i in range(0, self.max_iter):
            self.theta = self.theta - self.alpha * gradient(x, y.T[0], self.theta)
            self.loss_history[i] = self.loss_(y, self.predict_(x))
            # print(self.theta)
        self.thetas = np.array([[self.theta[0]], [self.theta[1]]])

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

    def mse_(self, y, y_hat):
        return np.square(np.subtract(y, y_hat)).mean()

    def predict(self, x):
        X = add_intercept(np.reshape(x, (len(x), 1)))
        return np.array([np.sum(X * self.theta, axis=1)])


def linear_regression():
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)
    mlr = MyLinearRegression([89, -8])
    y_hat = mlr.predict_(Xpill)
    mlr.fit_(Xpill, Yscore)
    plt.plot(Xpill, mlr.theta[0] + mlr.theta[1] * Xpill, '--', color='lime', marker='X')
    plt.plot(Xpill, Yscore, 'o', color='aqua')
    plt.plot(Xpill, y_hat, 'X', color='lime')
    plt.legend(['Spredict(pills)', 'Strue(pills)'], frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.1))
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.grid(True)
    plt.show()
    plot_cost(Xpill, Yscore)


def plot_cost(x, y):
    theta1 = np.linspace(-14, -4, 50)
    theta0 = np.linspace(74, 104, 6, endpoint=True)
    legend = []
    cpt = 0
    for t0 in theta0:
        loss = []
        for t1 in theta1:
            linear = MyLinearRegression(np.array([[t0], [t1]]))
            y_hatmp = linear.predict(x)
            loss.append(linear.loss_(y, y_hatmp))
        legend.append(f'J(θ0=c{cpt},θ1)')
        cpt += 1
        plt.plot(theta1, loss)
    plt.ylim(0, 150)
    plt.grid(True)
    plt.legend(legend)
    plt.show()


linear_regression()

# import numpy as np
# import matplotlib.pyplot as plt

# def square(x):
#     return x**2

# class MyLinearRegression():
#     """
#     Description:
#     My personnal linear regression class to fit like a boss.
#     """
#     def __init__(self, thetas, alpha=0.001, max_iter=1000):
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.thetasHistory = []
#         if thetas.ndim == 1:
#             self.thetas = np.array([[thetas[0]], [thetas[1]]], dtype=np.float64)
#         else:
#             self.thetas = thetas
#     @staticmethod
#     def test_type(values, type, size):
#         if isinstance(values, type):
#             if size > 0:
#                 if values.shape[0] != size:
#                     return False
#             elif values.shape[0] < 1:
#                 return False
#         else:
#             return False
#         return True

#     @staticmethod
#     def add_intercept(x):
#         if not MyLinearRegression.test_type(x, (np.ndarray, np.generic), 0):
#             return None
#         intercept = np.ones([x.shape[0], 1])
#         x = np.hstack((intercept, x))
#         return x

#     def predict_(self, x):
#         ret = []
#         if not MyLinearRegression.test_type(x, (np.ndarray, np.generic), 0):
#             return None
        
#         if x.ndim == 1: 
#             x = np.array([x]).T
        
#         x = MyLinearRegression.add_intercept(x)
#         if self.thetas.shape[0] > self.thetas.shape[1]:
#             self.thetas = self.thetas.T
        
#         ret = x * self.thetas
#         ret = np.sum(ret, axis=1)
#         if ret.ndim == 1:
#             ret = np.array([ret]).T
        
#         return ret
    
    
    
    
#     @staticmethod
#     def loss_elem_(y, y_hat):
#         if y.ndim == 1:
#             y = np.array([y]).T
#         if y_hat.ndim == 1:
#             y_hat = np.array([y_hat]).T
#         ret = y_hat - y
#         ret = np.array(list(map(square, ret)))
#         return ret
    
#     @staticmethod
#     def mse_(y, y_hat):
#         ret = MyLinearRegression.loss_elem_(y, y_hat)
#         ret = float(1/(len(y)) * sum(ret))
#         return ret

#     @staticmethod
#     def loss_(y, y_hat):
#         ret = MyLinearRegression.loss_elem_(y, y_hat)
#         ret = float(1/(2 * len(y)) * sum(ret))
#         return ret
    
#     @staticmethod
#     def gradient(x, y, theta):
#         if not MyLinearRegression.test_type(x, (np.ndarray, np.generic), 0):
#             return None
#         if not MyLinearRegression.test_type(y, (np.ndarray, np.generic), 0):
#             return None
#         if not MyLinearRegression.test_type(theta, (np.ndarray, np.generic), 0):
#             return None
        
#         y_transpose = y.T[0]
#         x_prime = MyLinearRegression.add_intercept(x)
#         nab = np.dot(x_prime, theta) - y_transpose
        
        
#         x_prime_transpose = x_prime.T

#         nab = np.dot(x_prime_transpose, nab)
#         nab = nab * (1 / len(y_transpose))

#         return nab

#     def fit_(self, x, y):
#         theta0 = self.thetas[0][0]
#         theta1 = self.thetas[0][1]
#         max_iter = self.max_iter
#         while(max_iter > 0):
#             nab = MyLinearRegression.gradient(x, y, np.array([theta0, theta1]))
#             nab0 = nab[0] * self.alpha
#             nab1 = nab[1] * self.alpha
#             theta0 = theta0 - nab0
#             theta1 = theta1 - nab1
#             self.thetasHistory.append(np.array([[theta0], [theta1]], dtype=np.float64))
#             max_iter -= 1
#         self.thetas[0][0] = theta0
#         self.thetas[0][1] = theta1
    
#     def plotH(self, x, y, y_hat):
#         plt.plot(x, self.thetas[0][0] + self.thetas[0][1] * x, color='#31cc32',
#          linestyle='dashed')
#         plt.plot(x, y, 'o', color='#00bfff')
#         plt.plot(x, y_hat, 'X', color='#31cc32')
        
#         plt.grid(True)
#         plt.legend(["Spredict(pills)", "Strue(pills)"], frameon=False,
#         loc='upper left', ncol=2 ,bbox_to_anchor=(0, 1.1))
#         plt.ylabel('Space driving score')
#         plt.xlabel('Quantity of blue pill (in micrograms)')
#         plt.show()

#     def plotJ(self, x, y):
#         theta1 = np.linspace(-14, -4, 50)
#         theta0 = np.linspace(74, 104, 6, endpoint=True)
#         legend = []
#         cpt = 0
#         for t0 in theta0:
#             loss = []
#             for t1 in theta1:
#                 linear = MyLinearRegression(np.array([[t0], [t1]]))
#                 y_hatmp = linear.predict_(x)
#                 loss.append(linear.loss_(y, y_hatmp))
#             legend.append(f'J(θ0=c{cpt},θ1)')
#             cpt += 1
#             plt.plot(theta1, loss)

#         plt.ylim(0,150)
#         plt.grid(True)
#         plt.legend(legend)
#         plt.show()