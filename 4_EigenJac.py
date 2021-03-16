#!/bin/python3

import numpy as np

from common.linearsystems import LinearSystem, HilbertSystem


eps = 1e-15

# Построение матрицы поворота T_ij
def Rotation(A, i, j):
    N = A.shape[0]

    x = -2 * A[i, j]
    y = A[i, i] - A[j, j]
    
    if np.abs(y) > eps: # y = 0 ?
        norm = np.sqrt(x * x + y * y)
        c = np.sqrt(0.5 + 0.5 * np.abs(y) / norm)
        s = 0.5 * np.sign(x * y) * np.abs(x) / (c * norm)
    else:
        c = s = 1 / np.sqrt(2)

    T = np.eye(N)
    T[i, i] = T[j, j] = c
    T[i, j], T[j, i] = -s, s
    return T


# Вычисление радиуса i-го круга Гершгорина
def GetR(A, i):
    return np.abs(A[i]).sum() - np.abs(A[i, i])


# Стратегия выбора максимального недиагонального элемента
def OffDiag(A):
    N = A.shape[0]

    # Вектор радиусов
    b = np.array([GetR(A, l) for l in range(N)])

    k = 1
    while b.max() > eps:
        absA = np.abs(A)
        i, j = 0, 1
        for l in range(N):
            for k in range(N):
                if l != k and absA[l, k] > absA[i, j]:
                    i, j = l, k
        #print('absA:', absA, sep='\n')
        #print('A[%d, %d] = %f' % (i + 1, j + 1, absA[i, j]))

        T = Rotation(A, i, j)
        A = T @ A @ T.T

        b = np.array([GetR(A, l) for l in range(N)])
        k += 1
    return A, k


# Стратегия выбора максимального элемента из строки с максимальным радиусом круга
def FromMaxRadius(A):
    N = A.shape[0]

    # Вектор радиусов
    b = np.array([GetR(A, l) for l in range(N)])
    i = np.argmax(b)

    k = 1
    while b[i] > eps:
        absA = np.abs(A[i])
        absA[0, i] = 0
        j = np.argmax(absA)

        T = Rotation(A, i, j)
        A = T @ A @ T.T

        #print('[%d, %d]' % (i + 1, j + 1))
        #print('A:', A, sep='\n')
        #print('b =', b)
        #b[i] = GetR(A, i)
        #b[j] = GetR(A, j)
        #print('b =', b)
        b = np.array([GetR(A, l) for l in range(N)])
        #print('b =', b)
        #input()

        k += 1
        i = np.argmax(b)
    return A, k


test_systems = [
    LinearSystem(
        'Диагональная матрица', A=(
        (5,   0,   0),
        (0, -10,   0),
        (0,   0,  13),
    )),
    LinearSystem(
        'Симметричная вещественная матрица', A=(
        (2,   6,   3),
        (6,  -4, -11),
        (3, -11,  15),
    )),
    HilbertSystem(6),
    HilbertSystem(15),
]

for mat in test_systems:
    print(mat.msg)
    print('A:', mat.A, sep='\n')
    print()

    print('Стратегия выбора максимального недиагонального элемента')
    A, k = OffDiag(mat.A)
    eigenVals = np.array([A[i, i] for i in range(A.shape[0])])
    print('Собственные числа:', eigenVals)
    print('Количество итераций:', k)
    print()

    print('Стратегия выбора максимального элемента из строки с максимальным радиусом круга')
    A, k = FromMaxRadius(mat.A)
    eigenVals = np.array([A[i, i] for i in range(A.shape[0])])
    print('Собственные числа:', eigenVals)
    print('Количество итераций:', k)

    print('=' * 80)
