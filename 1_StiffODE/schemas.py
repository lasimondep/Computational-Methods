import numpy as np
from scipy import optimize

from consts import *
from __bar import Bar


def _init(u0, t0, T, tau):
    M = len(u0)                                 # Количество уравнений в системе
    N = int(round((T - t0) / tau))              # Количество узлов в разбиении
    t = np.linspace(t0, t0 + N * tau, N + 1)    # Множество узлов в разбиении
    u = np.empty((N + 1, M))                    # Двумерный массив решений
    return M, N, t, u


# Явный метод Рунге-Кутты в 4 стадии
def RungeKutta(f, u0, t0, T, tau):
    M, N, t, u = _init(u0, t0, T, tau)

    u[0] = np.array(u0)

    print('\nRungeKutta, τ =', tau)
    _bar = Bar(N).start()

    for n in range(N):
        w1 = f(t[n], u[n])
        w2 = f(t[n] + tau / 2, u[n] + tau * w1 / 2)
        w3 = f(t[n] + tau / 2, u[n] + tau * w2 / 2)
        w4 = f(t[n] + tau / 2, u[n] + tau * w3)

        u[n + 1] = u[n] + tau * (w1 + 2 * w2 + 2 * w3 + w4) / 6
        _bar.update(n + 1)

    _bar.finish()
    return u, t


# Вычисление матрицы Якоби численным дифференцированием
def jac(f, t, u):
    M = len(u)
    res = np.empty((M, M))
    add = np.zeros(M)

    for k in range(M):
        add[k] = h_jac
        # f(..., u_k + h, ...) - f(..., u_k - h, ...)
        # -------------------------------------------
        #                      2h
        res[k] = (f(t, u + add) - f(t, u - add)) / (2 * h_jac)
        add[k] = 0
    return res.T


# Метод Розенброка с комплексными коэффициентами
def CROS1(f, u0, t0, T, tau):
    M, N, t, u = _init(u0, t0, T, tau)

    u[0] = np.array(u0)

    print('\nCROS1, τ =', tau)
    _bar = Bar(N).start()

    for n in range(N):
        # Линейное уравнение для w
        w = np.linalg.solve(np.eye(M) - a_CROS1 * tau * jac(f, t[n], u[n]),
                            f(t[n] + c_CROS1 * tau, u[n]))

        u[n + 1] = u[n] + b_CROS1 * tau * w.real
        _bar.update(n + 1)

    _bar.finish()
    return u, t


# Неявный метод Адамса в 4 шага
def Adams(f, u0, t0, T, tau):
    M, N, t, u = _init(u0, t0, T, tau)

    # Необходимо задать первые k_Adams значений решения u; + u0
    # Для этого используем метод Рунге-Кутты
    u[: k_Adams + 1] = RungeKutta(f, u0, t0, t[k_Adams], tau)[0]

    print('\nAdams, τ =', tau)
    _bar = Bar(N).start()

    w = np.empty((k_Adams + 1, M))
    for n in range(k_Adams, N):
        w.fill(0)

        # Нелинейное уравнение для u[n + 1]
        def equation(_u):
            w[0] = f(t[n + 1], _u)
            for l in range(1, k_Adams + 1):
                w[l] = f(t[n + 1 - l], u[n + 1 - l])
            return _u - u[n] - tau * b_Adams.dot(w)

        u[n + 1] = optimize.root(equation, u[n]).x
        _bar.update(n + 1)

    _bar.finish()
    return u, t
