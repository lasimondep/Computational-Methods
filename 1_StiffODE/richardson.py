import numpy as np


def Richardson(f, u0, t0, T, schema, p, tau, eps):
    M = len(u0)
    N = []
    DeltaL2 = []

    u, t = schema(f, u0, t0, T, tau) # Решение на первой сетке
    while True:
        tau /= 2 # Сгущение сетки
        u_new, t_new = schema(f, u0, t0, T, tau)
        N.append(len(t_new))

        # Погрешности по правилу Рунге
        delta = np.zeros((N[-1], M))
        delta[::2] = (u_new[::2] - u) / (2 ** p - 1)
        delta[1::2] = (delta[:-2:2] + delta[2::2]) / 2
        DeltaL2.append(np.sqrt((delta ** 2).sum() / N[-1])) # Норма погрешности
        print('|Δ| =', DeltaL2[-1])

        # Проверяем достижение заданной точности
        if eps > DeltaL2[-1] or N[-1] > 1e7:
            u_new += delta # Уточнение последнего решения
            return DeltaL2, N, u_new, t_new
        u, t = u_new, t_new
