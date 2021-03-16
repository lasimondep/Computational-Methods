import numpy as np


# Регуляризация эрмитовой матрицы
def regular_hermit(alpha, A, b, x0):
    return A + alpha * np.eye(A.shape[0]), b + alpha * x0


# Регуляризация произвольной матрицы
def regular_non_hermit(alpha, A, b, x0):
    return regular_hermit(alpha, A.H @ A, A.H.dot(b), x0)
