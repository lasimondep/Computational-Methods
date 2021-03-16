#!/bin/python3

import numpy as np

from common.linearsystems import LinearSystem, HilbertSystem


eps = 1e-10


# Степенной метод
def Power(A):
    x_prev = np.random.rand(A.shape[0])
    prev = np.full_like(x_prev, 1e8)

    k = 1
    while k < 1e5:
        x = np.ravel(A.dot(x_prev))
        now = np.abs(x / x_prev)
        norm = np.linalg.norm(x)
        if norm > 1e9:
            x_prev /= norm
            x /= norm
        if np.abs(prev - now).max() < eps:
            break
        prev = now
        x_prev = x
        k += 1
    return np.linalg.norm(x) / np.linalg.norm(x_prev), k


# Метод скалярных произведений
def DotProduct(A):
    x_prev = np.random.rand(A.shape[0])
    y_prev = np.random.rand(A.shape[0])
    prev = 1e8

    k = 1
    while k < 1e5:
        x = np.ravel(A.dot(x_prev))
        y = np.ravel(A.T.dot(y_prev))
        now = np.abs(x.dot(y) / x_prev.dot(y))
        norm = np.linalg.norm(x)
        if norm > 1e9:
            x_prev /= norm
            x /= norm
            y /= norm
        if np.abs(prev - now) < eps:
            break
        prev = now
        x_prev = x
        y_prev = y
        k += 1
    return x.dot(y) / x_prev.dot(y), k


test_systems = [
    LinearSystem(
        'Диагональная матрица', A=(
        (5,   0,   0),
        (0, -10,   0),
        (0,   0,  13),
    )),
    LinearSystem(
        'Симметричная вещественная матрица', A=np.eye(5)
    ),
    LinearSystem(
        'Симметричная вещественная матрица с нулевыми элементами', A=(
        (2,   0,   3),
        (0,  -4, -11),
        (3, -11,  15),
    )),
    LinearSystem(
        'Несимметричная матрица', A=(
        (13, 7,   4),
        (1,  2, -17),
        (2,  8,  -9),
    )),
    HilbertSystem(25),
    HilbertSystem(100),
]

for i in range(5):
    for j in range(i):
        test_systems[1].A[i, j] = i
        test_systems[1].A[j, i] = i

for mat in test_systems:
    print(mat.msg)
    print('True max eigen                          :', np.linalg.eigvals(mat.A).max())
    print()

    print('Степенной метод')
    eigenVal, k = Power(mat.A)
    print('Максимальное по модулю собственное число:', eigenVal)
    print('Количество итераций:', k)
    print()

    print('Степенной метод')
    eigenVal, k = DotProduct(mat.A)
    print('Максимальное по модулю собственное число:', eigenVal)
    print('Количество итераций:', k)
    print('=' * 80)
