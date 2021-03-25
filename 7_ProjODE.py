#!/bin/python3

import numpy as np
from scipy.integrate import quad
import math
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

def Galerkin(p, q, r, f, a, b, alpha, beta, N):
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

    c = np.linalg.solve(A, d)
    return lambda x: u(phi, c, x)

def Colocation(p, q, r, f, a, b, alpha, beta, N):
    t = np.array([np.cos((2 * k - 1) * np.pi / (2 * N)) for k in range(1, N + 1)]) # В качестве узлов колокации --- корни многочлена Чебышева
	
    f = f(t)
    phi = []
    phiDeriv = []
    phiDeriv2 = []

    A = np.matrix(np.zeros((N, N)))
    
    for i in range(N):
        phi.append(lambda x, i: (1 - x * x) * PolyJac(1, i, x))
        phiDeriv.append(lambda x, i: PolyJacPhiDeriv(1, i, x))
        phiDeriv2.append(lambda x, i: PolyJacPhiDeriv2(1, i, x))
    
    for j in range(N):
        for i in range(N):
            A[j, i] = q(t[j]) * phiDeriv[i](t[j], i) + p(t[j])*phiDeriv2[i](t[j], i) + r(t[j]) * phi[i](t[j], i) * (1 - t[j] * t[j])
    
    c = np.linalg.solve(A, f)
    return lambda x: u(phi, c, x)

def Richardson(u, u_prev, p=2):
    # Погрешности по правилу Рунге
    t = np.linspace(-1, 1)
    delta = np.linalg.norm((u(t)  - u_prev(t)) * (2 ** p) / (2 ** p - 1))
    return delta

class ODE:
    """
    Общий вид уравнения:
    pu''(x) + qu'(x) + ru(x) = f(x)
    граничные условия:
    alpha[0]u(a) - alpha[1]u'(a) = alpha[2]
    beta[0]u(b) +  beta[1]u'(b) =  beta[2]
    """
    def __init__(self, msg, p, q, r, f, a, b, alpha, beta, N):
        self.msg = msg
        self.p = p
        self.q = q
        self.r = r
        self.f = f
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.N = N

test_case = [
    (ODE('Вариант 3',
        p=lambda x: -1 / (x - 3),
        q=lambda x: 1 + x / 2,
        r=lambda x: np.exp(x / 2),
        f=lambda x: 2 - x,
        a=-1, b=1,
        alpha=(1, 0, 0), beta=(1, 0, 0),
        N=[4, 6, 8, 10]),
    1e-5),
    (ODE('Вариант 6',
        p=lambda x: (x - 2) / (x + 2),
        q=lambda x: x,
        r=lambda x: 1 - np.sin(x),
        f=lambda x: x * x,
        a=-1, b=1,
        alpha=(1, 0, 0), beta=(1, 0, 0),
        N=[4, 6, 8, 10]),
    1e-7),
    (ODE('Вариант 8',
        p=lambda x: -(4 - x) / (5 - 2 * x),
        q=lambda x: (1 - x) / 2,
        r=lambda x: np.log(3 + x) / 2,
        f=lambda x: 1 + x / 3,
        a=-1, b=1,
        alpha=(1, 0, 0), beta=(1, 0, 0),
        N=[4, 6, 8, 10]),
    1e-8),
    (ODE('Вариант 16',
        p=lambda x: 1 + 0 * x,
        q=lambda x: -np.cos(x) / (2 + x),
        r=lambda x: 2 - x,
        f=lambda x: x + 1,
        a=-1, b=1,
        alpha=(0.2, -1, -0.8), beta=(0.9, 1, -0.1),
        N=[4, 6, 8, 10]),
    1e-8),
]

for ode, eps in test_case:
    print(ode.msg)
    m = len(ode.N)
    uGalerkin = []
    uGalerkin.append(Galerkin(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, 2))
    delta1 = np.zeros(m)
    uColocation = []
    uColocation.append(Colocation(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, 2))
    delta2 = np.zeros(m)
    for i in range(m):
         uGalerkin.append(Galerkin(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, ode.N[i]))
         delta1[i] = Richardson(uGalerkin[i + 1], uGalerkin[i])
         uColocation.append(Colocation(ode.p, ode.q, ode.r, ode.f, ode.a, ode.b, ode.alpha, ode.beta, ode.N[i]))
         delta2[i] = Richardson(uColocation[i + 1], uColocation[i])
    _x = np.linspace(-1, 1)
    plot_xuGalerkin = plt.Plot2D((r'$x$', r'$u$'))
    for i in range(m):
    	plot_xuGalerkin.plot(_x, uGalerkin[i + 1](_x), '-', label="%d nodes" % (ode.N[i]))

    plot_xuColocation = plt.Plot2D((r'$x$', r'$u$'))
    for i in range(m):
    	plot_xuColocation.plot(_x, uColocation[i + 1](_x), '-', label="%d nodes" % (ode.N[i]))

    plot_Delta = plt.Plot2D((r'$x$', r'\Delta'), (False, True))
    plot_Delta.plot(ode.N, delta1, '-', label="Galerkin")
    plot_Delta.plot(ode.N, delta2, '-', label="Colocation")
    plot_Delta.show()
