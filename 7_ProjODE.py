#!/bin/python3

import numpy as np
from scipy.integrate import quad

import common.__plots as plt


def PolyJac(k, n, x):
    if n == 0:
        return 1
    if n == 1:
        return (k + 1) * x
    return (n + k) * ((2 * n + 2 * k - 1) * x * PolyJac(k, n - 1, x) - (n + k - 1) * PolyJac(k, n - 2, x)) / (n + 2 * k) / n

def PolyJacDeriv(k, n, x):
    if n == 0:
        return 0
    return (n + 2 * k + 1) * PolyJac(k + 1, n - 1, x) / 2

def PolyJacDeriv2(k, n, x):
    if n == 0:
        return 0
    return (n + 2 * k + 1) * PolyJacDeriv(k + 1, n - 1, x) / 2

def PolyJacPhiDeriv(k, n, x):
    if k == 0:
        return PolyJacDeriv(k, n, x)
    return -2 * (n + 1) * (1 - x * x) ** (k - 1) * PolyJac(k - 1, n + 1, x)

def PolyJacPhiDeriv2(k, n, x):
    return -2 * (n + 1) * PolyJacPhiDeriv(k - 1, n + 1, x)

def u(phi, c, x):
    res = 0
    for n, it in enumerate(phi):
        res += it(x, n) * c[n]
    return res

def Galerkin(p, q, r, f, a, b, alpha, beta):
    N = 4
    phi = []
    phiDeriv = []
    phiDeriv2 = []

    for i in range(N):
        phi.append(lambda x, i: (1 - x * x) * PolyJac(1, i, x))
        phiDeriv.append(lambda x, i: PolyJacPhiDeriv(1, i, x))
        phiDeriv2.append(lambda x, i: PolyJacPhiDeriv2(1, i, x))

    A = np.matrix(np.zeros((N, N)))
    d = np.zeros(N)
    for i in range(N):
        d[i] = quad(lambda x, i: f(x) * phi[i](x, i), a, b, args=(i))[0]
        for j in range(N):
            A[i, j] = quad(lambda x, i, j: (p(x) * phiDeriv2[j](x, j) + q(x) * phiDeriv[j](x, j) + r(x) * phi[j](x, j)) * phi[i](x, i), a, b, args=(i, j))[0]

    print('A:', A)
    print('d:', d)
    c = np.linalg.solve(A, d)
    print('c:', c)
    return lambda x: u(phi, c, x)


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
    #(ODE('Вариант 3',
    #    p=lambda x: -1 / (x - 3),
    #    q=lambda x: 1 + x / 2,
    #    r=lambda x: np.exp(x / 2),
    #    f=lambda x: 2 - x,
    #    a=-1, b=1,
    #    alpha=(1, 0, 0), beta=(1, 0, 0)),
    #1e-5),
    #(ODE('Вариант 6',
    #    p=lambda x: (x - 2) / (x + 2),
    #    q=lambda x: x,
    #    r=lambda x: 1 - np.sin(x),
    #    f=lambda x: x * x,
    #    a=-1, b=1,
    #    alpha=(1, 0, 0), beta=(1, 0, 0)),
    #1e-7),
    (ODE('Вариант 8',
        p=lambda x: -(4 - x) / (5 - 2 * x),
        q=lambda x: (1 - x) / 2,
        r=lambda x: np.log(3 + x) / 2,
        f=lambda x: 1 + x / 3,
        a=-1, b=1,
        alpha=(1, 0, 0), beta=(1, 0, 0)),
    1e-8),
    #(ODE('Вариант 16',
    #    p=lambda x: 1 + 0 * x,
    #    q=lambda x: -np.cos(x) / (1 + x),
    #    r=lambda x: 2 - x,
    #    f=lambda x: x + 1,
    #    a=0, b=1,
    #    alpha=(0.2, -1, -0.8), beta=(0.9, 1, -0.1)),
    #1e-8),
]

for ode, eps in test_case:
    print(ode.msg)
    _u = Galerkin(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta)
    _x = np.linspace(-1, 1)
    plot_xu = plt.Plot2D((r'$x$', r'$u$'))
    plot_xu.plot(_x, _u(_x), '-')
    plot_xu.show()

    #x, u, N, DeltaL2 = Richardson(ode, eps=eps)

    #plot_xu = plt.Plot2D((r'$x$', r'$u$'))
    #plot_NDelta = plt.Plot2D((r'$N$', r'$\Delta$'), (False, True))

    #plot_xu.plot(x, u, '-')
    #plot_NDelta.plot(N, DeltaL2, '-')

    #plot_xu.show()
