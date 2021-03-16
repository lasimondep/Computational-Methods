#!/bin/python3

import numpy as np

from common.linearsystems import LinearSystem


# Построение матрицы B и вектора c
# для итерационных одношаговых методов
def GetBc(A, b):
    N = A.shape[0]
    B = np.matrix(A, dtype=np.float64)
    c = np.array(b, dtype=np.float64)

    for n in range(N):
        c[n] /= B[n, n]
        B[n] /= -B[n, n]
        B[n, n] = 0
    
    return B, c


# Метод простых итераций
def SimpleIter(A, b, eps, x_prev=None):
    N = A.shape[0]
    B, c = GetBc(A, b)
    #print('B:')
    #print(B)
    if x_prev is None:
        x_prev = np.zeros(N)

    k = 1
    while True:
        if k > 1e4:
            print('Расхождение метода простой итерации')
            return x, k
        x = np.ravel(B.dot(x_prev)) + c
        if np.linalg.norm(x - x_prev) < eps:
            return x, k
        k += 1
        x_prev = x


# Метод Зейделя
def Seidel(A, b, eps, x_prev=None):
    N = A.shape[0]
    B, c = GetBc(A, b)
    if x_prev is None:
        x_prev = np.zeros(N)

    k = 1
    x = np.zeros(N)
    while True:
        if k > 1e4:
            print('Расхождение метода Зейделя')
            return x, k
        for n in range(N):
            x[n] = B[n, :n].dot(x[:n]) + B[n, n + 1:].dot(x_prev[n + 1:]) + c[n]
        if np.linalg.norm(x - x_prev) < eps:
            return x, k
        k += 1
        x_prev = x.copy()


test_case = [
    LinearSystem(
        'Тест 1: матрица со строгим диагональным преобладанием', A=(
        (239, -19,  23),
        (  7, -97,   3),
        (-17,  11, 139),
    ), b=(3, -19, 7)),
    LinearSystem(
        'Тест 2: диагональное преобладание не строгое', A=(
        (-19,  6, -5),
        ( -2, 23,  1),
        (  0,  5, 19),
    ), b=(2, 0, -9)),
    LinearSystem(
        'Тест 3: диагональное преобладание отсутствует', A=(
        (-28, 7,  -9),
        (3,   7,   4),
        (1,  -2, -11),
    ), b=(-15, 0, 1)),
    LinearSystem(
        'Тест 4: все диагональные элементы меньше суммы остальных', A=(
        (-13, 9,  -8),
        (4,   7,   5),
        (-5,  -2, 11),
    ), b=(-5, 10, 3)),
    LinearSystem(
        'Тест 5: расходящаяся система', A=(
        ( 2, -8,  5),
        ( 3, -4, -2),
        (-6,  1,  2),
    ), b=(-1, 1, -1)),
]

eps = 1e-15

for lae in test_case:
    print(lae.msg)
    print('A:', lae.A, sep='\n')
    print('b =', lae.b)
    print('Метод простых итераций')
    x, k = SimpleIter(lae.A, lae.b, eps)
    print('x:', x)
    print('Количество итераций:', k)
    print('Метод Зейделя')
    x, k = Seidel(lae.A, lae.b, eps)
    print('x:', x)
    print('Количество итераций:', k)
    print()
