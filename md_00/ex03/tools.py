import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
    x as a numpy.array, a vector of shape m * 2.
    None if x is not a numpy.array.
    None if x is a empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.array) and len(x) == 0:
        return None
    return np.hstack((np.ones((len(x), 1)), x))


# # Example 1:
# x = np.arange(1,6).reshape((5,1))
# print(add_intercept(x))
# # Output:
# # array([[1., 1.],
# # [1., 2.],
# # [1., 3.],
# # [1., 4.],
# # [1., 5.]])

# # Example 2:
# y = np.arange(1,10).reshape((3,3))
# print(add_intercept(y))
# # Output:
# # array([[1., 1., 2., 3.],
# # [1., 4., 5., 6.],
# # [1., 7., 8., 9.]])