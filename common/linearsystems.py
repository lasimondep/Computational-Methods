import numpy as np

class LinearSystem:
    def __init__(self, msg, A, b=None, x=None):
        self.msg = msg
        self.A = np.matrix(A, dtype=np.float64)
        self.b = b
        self.x = x


class HilbertSystem(LinearSystem):
    def __init__(self, N, **kwargs):
        A = np.matrix(np.empty((N, N)))
        for n in range(N):
            for m in range(N):
                A[n, m] = 1 / (n + m + 1)
        e = np.ones(N, dtype=np.float64)
        super().__init__(
            'Матрица Гилберта %dx%d' % (N, N),
            A=A, b=np.ravel(A.dot(e)), x=e, **kwargs)
