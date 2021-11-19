def is_vector(x):
    if not isinstance(x, list):
        return False
    for elt in x:
        if not isinstance(elt, int) and not isinstance(elt, float):
            return False
    return True


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector(x):
        return None
    if not isinstance(theta, np.array) and len(theta) != 2:
        return None
    return [float(theta[0] + theta[1] * xi) for xi in x]

# import numpy as np
# x = np.arange(1,6)
# # Example 1:
# theta1 = np.array([5, 0])
# print(simple_predict(x, theta1))
# # Ouput:
# # array([5., 5., 5., 5., 5.])

# # Example 2:
# theta2 = np.array([0, 1])
# print(simple_predict(x, theta2))
# # Output:
# # array([1., 2., 3., 4., 5.])

# # Example 3:
# theta3 = np.array([5, 3])
# print(simple_predict(x, theta3))
# # Output:
# # array([ 8., 11., 14., 17., 20.])

# # Example 4:
# theta4 = np.array([-3, 1])
# print(simple_predict(x, theta4))
# # Output:
# # array([-2., -1., 0., 1., 2.])