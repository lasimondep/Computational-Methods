#!/bin/python3

import numpy as np

import common.__plots as plt


def GridFDM(p, q, r, f, a, b, alpha, beta, N):
    h = (b - a) / N
    x = np.linspace(a - h / 2, b + h / 2, N + 2) # Сдвинутая сетка
    p, q, r, f = p(x), q(x), r(x), f(x)

    A = np.zeros((N + 2, N + 2))
    b = np.zeros(N + 2)
    A[0, 0]         = alpha[0] / 2 + alpha[1] / h
    A[0, 1]         = alpha[0] / 2 - alpha[1] / h
    A[N + 1, N]     =  beta[0] / 2 -  beta[1] / h
    A[N + 1, N + 1] =  beta[0] / 2 +  beta[1] / h
    b[0]    = alpha[2]
    b[N + 1] = beta[2]
    for n in range(1, N + 1):
        A[n, n - 1] = (p[n] / h - q[n] / 2) / h
        A[n, n] = r[n] - 2 * p[n] / h / h
        A[n, n + 1] = (p[n] / h + q[n] / 2) / h
        b[n] = f[n]

    return x, np.linalg.solve(A, b)


class ODE:
    """
    Общий вид уравнения:
    pu''(x) + qu'(x) + ru(x) = f(x)
    граничные условия:
    alpha[0]u(a) - alpha[1]u'(a) = alpha[2]
    beta[0]u(b) +  beta[1]u'(b) =  beta[2]
    """
    def __init__(self, msg, p, q, r, f, a, b, alpha, beta):
        self.msg = msg
        self.p = p
        self.q = q
        self.r = r
        self.f = f
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta


def Richardson(ode, p=2, N=10, eps=1e-5):
    DeltaL2 = []
    _N = [N,]
    x_prev, u_prev = GridFDM(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, N)
    while N < 1e4:
        N *= 2 # Сгущение сетки
        print('N =', N)
        x, u = GridFDM(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, N)

        # Погрешности по правилу Рунге
        delta = ((u[1:-1:2] + u[2::2]) / 2 - u_prev[1:-1]) * (2 ** p) / (2 ** p - 1)
        DeltaL2.append(np.sqrt((delta ** 2).sum() / delta.shape[0]))
        print('|Δ| =', DeltaL2[-1])

        # Проверяем достижение заданной точности
        if eps > DeltaL2[-1]:
            u_prev[1:-1] += delta
            break
        x_prev, u_prev = x, u
        _N.append(N)

    return x_prev, u_prev, _N, DeltaL2


test_case = [
    (ODE('Вариант 3',
        p=lambda x: -1 / (x - 3),
        q=lambda x: 1 + x / 2,
        r=lambda x: np.exp(x / 2),
        f=lambda x: 2 - x,
        a=-1, b=1,
        alpha=(1, 0, 0), beta=(1, 0, 0)),
    1e-5),
    (ODE('Вариант 5',
        p=lambda x: -1 / (x + 3),
        q=lambda x: -x,
        r=lambda x: np.log(2 + x),
        f=lambda x: 1 - x / 2,
        a=-1, b=1,
        alpha=(0, 1, 0), beta=(0.5, 1, 0)),
    1e-7),
    (ODE('Вариант 7',
        p=lambda x: -(4 + x) / (5 + 2 * x),
        q=lambda x: x / 2 - 1,
        r=lambda x: 1 + np.exp(x / 2),
        f=lambda x: 2 + x,
        a=-1, b=1,
        alpha=(0, 1, 0), beta=(1, 2, 0)),
    1e-8),
    (ODE('Вариант 16',
        p=lambda x: 1 + 0 * x,
        q=lambda x: -np.cos(x) / (1 + x),
        r=lambda x: 2 - x,
        f=lambda x: x + 1,
        a=0, b=1,
        alpha=(0.2, -1, -0.8), beta=(0.9, 1, -0.1)),
    1e-8),
]

for ode, eps in test_case:
    print(ode.msg)
    x, u, N, DeltaL2 = Richardson(ode, eps=eps)

    plot_xu = plt.Plot2D((r'$x$', r'$u$'))
    plot_NDelta = plt.Plot2D((r'$N$', r'$\Delta$'), (False, True))

    plot_xu.plot(x, u, '-')
    plot_NDelta.plot(N, DeltaL2, '-')

    plot_xu.show()
