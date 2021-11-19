import numpy as np


def is_vector(x):
    if not isinstance(x, list):
        return False
    for elt in x:
        if not isinstance(elt, int) and not isinstance(elt, float):
            return False
    return True


class TinyStatistician:
    def __init__(self):
        pass

    @staticmethod
    def mean(x):
        if not is_vector(x):
            return None
        ret = 0
        for elt in x:
            ret += elt
        return float(ret / len(x))

    @staticmethod
    def median(x):
        if not is_vector(x):
            return None
        len_x = len(x)
        x.sort()
        if len_x % 2 == 0:
            return float((x[len_x / 2] + x[(len_x / 2) - 1]) / 2)
        else:
            return float(x[len_x // 2])

    @staticmethod
    def quartile(x):
        if not is_vector(x):
            return None
        len_x = len(x)
        x.sort()
        if len_x % 2 == 0:
            return [float(x[len_x / 4 - 1]), float(x[len_x / 4 * 3 - 1])]
        else:
            return [float(x[int(len_x / 4)]), float(x[int(len_x / 4 * 3)])]

    @staticmethod
    def percentile(x, p):
        if not is_vector(x):
            return None
        if not isinstance(p, int) and not isinstance(p, float):
            return None
        i = p * len(x) / 100
        if int(i) != 0 and i % int(i) == 0:
            return float(x[i - 1])
        return float(x[int(i)])

    @staticmethod
    def var(x):
        if not is_vector(x):
            return None
        m = sum(x) / len(x)
        return sum((xi - m) ** 2 for xi in x) / len(x)

    @staticmethod
    def std(x):
        if not is_vector(x):
            return None
        m = sum(x) / len(x)
        return np.sqrt(sum((xi - m) ** 2 for xi in x) / len(x))

# a = [1, 42, 300, 10, 59]
# stat = TinyStatistician()
# print(stat.mean(a))
# # Output:
# # 82.4
# print(stat.median(a))
# # Output:
# # 42.0
# print(stat.quartile(a))
# # Output:
# # [10.0, 59.0]
# print(stat.percentile(a, 10))
# # Output:
# # 1.0
# print(stat.percentile(a, 28))
# # Output:
# # 10.0
# print(stat.percentile(a, 83))
# # Output:
# # 300.0
# print(stat.var(a))
# # Output:
# # 12279.4399...
# print(stat.std(a))
# # Output:
# # 110.8126