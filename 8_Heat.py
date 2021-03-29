#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from common.__plots import Plot3D

np.set_printoptions(precision=10, linewidth=1e8, suppress=True, floatmode='fixed')

class HeatEquation:
    """
    Общий вид уравнения:
    u_t(x, t) - kappa * u_xx(x, t) = f(x, t)
    t in [0, T], x in [a, b]
    начальное условие:
    u(x, 0) = phi(x)
    граничные условия:
    u(a, t) = alpha(t)
    u(b, t) =  beta(t)
    """
    def __init__(self, msg, u, kappa, f, T, phi, a, b, alpha, beta):
        self.msg = msg
        self.u = u
        self.kappa = kappa
        self.f = f
        self.T = T
        self.phi= phi
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta

test_case = [
    HeatEquation('Тест 1',
        u=lambda x, t: x * x * x * t * t * t,
        kappa=1,
        f=lambda x, t: 3 * x * x * x * t * t - 6 * x * t * t * t,
        T=0.1,
        phi=lambda x: 0 * x,
        a=0, b=1,
        alpha=lambda t: -t * t * t,
        beta=lambda t: t * t * t
        ),
    HeatEquation('Тест 2',
        u=lambda x, t: np.sin(1 + 2 * t) * np.cos(2 * x),
        kappa=2,
        f=lambda x, t: (2 * np.cos(1 + 2 * t) + 8 * np.sin(1 + 2 * t)) * np.cos(2 * x),
        T=0.1,
        phi=lambda x: np.sin(1) * np.cos(2 * x),
        a=-0.5, b=1,
        alpha=lambda t: np.cos(1) * np.sin(1 + 2 * t),
        beta=lambda t: np.cos(2) * np.sin(1 + 2 * t)
        ),
]

def Explicit(kappa, f, T, phi, a, b, alpha, beta, N, M):
    # Делим [a, b] на сетку h
    x, h = np.linspace(a, b, N + 1, retstep=True)

    # Делим [0, T] на сетку tau
    t, tau = np.linspace(0, T, M + 1, retstep=True)

    u = np.zeros((M + 1, N + 1))

    # Заполняем начальный слой
    u[0] = phi(x)

    # Вычисляем следующие слои
    for k in range(1, len(t)):
        for n in range(1, N):
            u[k, n] = u[k - 1, n] + tau * (kappa * (u[k - 1, n - 1] - 2 * u[k - 1, n] + u[k - 1, n + 1]) / h / h + f(x[n], t[k]))
        # Граничные точки
        u[k, 0] = alpha(t[k])
        u[k, -1] = beta(t[k])
    return x, t, u

def HeatSolve(kappa, f, T, phi, a, b, alpha, beta, sigma, N, M):
    # Делим [a, b] на сетку h
    x, h = np.linspace(a, b, N + 1, retstep=True)
    print('h =', h)

    # Делим [0, T] на сетку tau
    t, tau = np.linspace(0, T, M + 1, retstep=True)
    print('tau =', tau)

    u = np.zeros((M + 1, N + 1))

    # Заполняем начальный слой
    u[0] = phi(x)

    # Матрица Λ
    Lambda = np.zeros((N + 1, N + 1))
    #for n in range(N - 1):
    for n in range(N + 1):
        Lambda[n, n] = -2 * kappa / h / h
        if n > 0:
            Lambda[n, n - 1] = kappa / h / h
        if n < N:
            Lambda[n, n + 1] = kappa / h / h

    # Вычисляем следующие слои
    A = csc_matrix(np.eye(N + 1) - sigma * tau * Lambda)
    for k in range(1, len(t)):
        # Граничные точки
        u[k, 0] = alpha(t[k])
        u[k, -1] = beta(t[k])

        F = Lambda.dot(u[k - 1]) + f(x, t[k])
        w = spsolve(A, F)
        u[k] = u[k - 1] + tau * w.real


        u[k, 0] = alpha(t[k])
        u[k, -1] = beta(t[k])

    return x, t, u


for heatEquation in test_case:
    print(heatEquation.msg)

    explicit_plot = Plot3D(('$x$', '$t$', '$\\tilde u$'))
    explicit_delta_plot = Plot3D(('$x$', '$t$', '$|u(x, t) - \\tilde u|$'))
    implicit_plot = Plot3D(('$x$', '$t$', '$\\tilde u$'))
    implicit_delta_plot = Plot3D(('$x$', '$t$', '$|u(x, t) - \\tilde u|$'))

    print('Явная схема σ = 0')
    #x, t, u = Explicit(heatEquation.kappa, heatEquation.f, heatEquation.T, heatEquation.phi, heatEquation.a, heatEquation.b, heatEquation.alpha, heatEquation.beta, 100, 10000)
    x, t, u = HeatSolve(heatEquation.kappa, heatEquation.f, heatEquation.T, heatEquation.phi, heatEquation.a, heatEquation.b, heatEquation.alpha, heatEquation.beta, 0, 100, 5000)
    u_real = np.zeros_like(u)
    for k in range(len(t)):
        u_real[k] = heatEquation.u(x, t[k])
    x, t = np.meshgrid(x, t)
    explicit_plot.plot(x, t, u)
    explicit_delta_plot.plot(x, t, np.abs(u_real - u))

    print('Неявная схема σ = (1 + i) / 2')
    x, t, u = HeatSolve(heatEquation.kappa, heatEquation.f, heatEquation.T, heatEquation.phi, heatEquation.a, heatEquation.b, heatEquation.alpha, heatEquation.beta, 0.5 + 0.5j, 100, 5000)
    u_real = np.zeros_like(u)
    for k in range(len(t)):
        u_real[k] = heatEquation.u(x, t[k])
    x, t = np.meshgrid(x, t)
    implicit_plot.plot(x, t, u)
    implicit_delta_plot.plot(x, t, np.abs(u_real - u))

    explicit_plot.show()
