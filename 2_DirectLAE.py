#!/bin/python3

import numpy as np

from common.linearsystems import HilbertSystem
from common.decompositions import LU, QR
from common.regularization import regular_hermit
from common.__plots import Plot2D
from common.__bar import Bar


# Решение системы Ax = b с нижней диагональной матрицей
def Lsolve(A, b):
    N = A.shape[0]
    x = np.zeros(N)
    for n in range(N):
        x[n] = (b[n] - A[n].dot(x)) / A[n, n]
    return x


test_case = [
    HilbertSystem(25),
    HilbertSystem(31),
    HilbertSystem(40),
]

# Параметры регуляризации
Alpha = np.arange(1, 1e4) * 1e-5

for _i, lae in enumerate(test_systems):
    Delta_LU = []
    Delta_QR = []

    print(lae.msg)
    print('lae.x:', lae.x)

    L, U = LU(lae.A)
    x = Lsolve(U[::-1, ::-1], Lsolve(L, lae.b)[::-1])[::-1]
    print('LU разложение:')
    print('|Δ| =', np.linalg.norm(x - lae.x))
    print('x:', x)

    Q, R = QR(lae.A)
    x = Lsolve(U[::-1, ::-1], np.ravel(Q.dot(lae.b))[::-1])[::-1]
    print('QR разложение:')
    print('|Δ| =', np.linalg.norm(x - lae.x))
    print('x:', x)
    
    _bar = Bar(len(Alpha)).start()
    for _j, alpha in enumerate(Alpha):
        A, b = regular_hermit(alpha, lae.A, lae.b, lae.x)

        L, U = LU(A)
        x = Lsolve(U[::-1, ::-1], Lsolve(L, b)[::-1])[::-1]
        Delta_LU.append(np.linalg.norm(x - lae.x))

        Q, R = QR(A)
        x = Lsolve(R[::-1, ::-1], np.ravel(Q.dot(b))[::-1])[::-1]
        Delta_QR.append(np.linalg.norm(x - lae.x))

        _bar.update(_j + 1)

    _bar.finish()

    plotLU = Plot2D((r'$\alpha$', r'$|\Delta|$'), (True, False))
    plotLU.plot(Alpha, Delta_LU, '.', label='LU decomposition')

    plotQR = Plot2D((r'$\alpha$', r'$|\Delta|$'), (True, False))
    plotQR.plot(Alpha, Delta_QR, '.', label='QR decomposition')
    
    plotLU.save('plotLU_%d.png' % (_i + 1))
    plotQR.save('plotQR_%d.png' % (_i + 1))
