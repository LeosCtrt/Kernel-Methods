import numpy as np


def kernel_exp(x, alpha):
    return np.exp(alpha * (x - 1))


def kernel_poly(x, alpha=2):
    return np.power(x, alpha)


KERNELS = {"exp": kernel_exp, "poly": kernel_poly}
