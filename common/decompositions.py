import numpy as np


# Разложение A = LU
def LU(_A):
    # Копируем, чтобы не изменять исходную матрицу
    # и гарантировать, что type(A) = np.matrix
    A = np.matrix(_A, dtype=np.float64)
    N = A.shape[0]
    L = np.matrix(np.eye(N))

    for n in range(N - 1):
        # Строим матрицу специального вида
        M = np.matrix(np.eye(N))
        M[n + 1:, n] = A[n + 1:, n] / A[n, n]

        L = L @ M
        M[n + 1:, n] *= -1
        A = M @ A

    return L, A


# Разложение A = QR методом отражений
def QR(_A):
    eps = 1e-16
    # Копируем, чтобы не изменять исходную матрицу
    # и гарантировать, что type(A) = np.matrix
    A = np.matrix(_A, dtype=np.float64)
    N = A.shape[0]
    Q = np.matrix(np.eye(N))

    for n in range(N):
        w = A[n:, n].copy()       # w = a_n = (a_1n, ..., a_nn).T 
        w[0] -= np.linalg.norm(w) # w = a_n - |a_n|e 

        w_norm = np.linalg.norm(w)
        if w_norm > eps: # |w| = 0 => w = 0 => H = E
            w /= w_norm

        # Матрица отражения Хаусхолдера
        H = np.matrix(np.eye(N - n)) - 2 * w @ w.H

        Q[n:, n:] = Q[n:, n:] @ H
        A[n:, n:] = H @ A[n:, n:]

    return Q, A
