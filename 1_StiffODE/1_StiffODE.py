#!/bin/python3

import numpy as np

from schemas import RungeKutta, CROS1, Adams
from richardson import Richardson
from odesystems import SimpleSystem, LinearSystem, AutonomousSystem, ConstEigenSystem, NonLinearSystem


test_case = [
    SimpleSystem(a=1, b=-4,
        u0=(0.8, 1),
        t0=-1, T=1),
    LinearSystem(a=-1, b=-10000,
        u0=(1, 1, 1000, 1000, 1000, 1000),
        t0=0, T=1),
    AutonomousSystem(a=1, b=100, c=-1000,
        u0=(5, 0.001),
        t0=-2, T=2),
    ConstEigenSystem(a=-50, b=60, c=59,
        u0=(10, 0.01),
        t0=0, T=1),
    NonLinearSystem(a=1, b=5, c=-300,
        u0=(2, 0.01),
        t0=0, T=1),
]

schemas = (
    # Метод  Теор порядок  Начальный шаг  Требуемая
    #          точности        сетки       точность
    (RungeKutta,   4,           1e-2,        1e-8),
    (CROS1,        2,           1e-2,        1e-6),
    (Adams,        4,           1e-2,        1e-8),
)

import time
_ts = time.time()
for ode in test_case:
    print(ode.msg)
    with open('out.txt', 'a') as fout:
        print(ode.msg, file=fout)
    for schema, p, tau, eps in schemas:
        with open('out.txt', 'a') as fout:
            print(schema.__name__, file=fout)
            print("\tN\t|Δ|", file=fout)
        Delta, N, u, t = Richardson(ode.f, ode.u0, ode.t0, ode.T, schema, p, tau, eps)
        with open('out.txt', 'a') as fout:
            for _N, _Delta in zip(N, Delta):
                print('\t%d\t%f' % (_N, _Delta), file=fout)
        with open('./csv/NDelta_' + ode.__class__.__name__ + '_' + schema.__name__ + '.csv', 'a') as fout:
            for _N, _Delta in zip(N, Delta):
                print('%d,%.16f' % (_N, _Delta), file=fout)
        with open('./csv/tu_' + ode.__class__.__name__ + '_' + schema.__name__ + '.csv', 'a') as fout:
            for n in range(N[-1]):
                print('%.16f,' % t[n], end='', file=fout)
                for it in u[n]:
                    print(it, end=',', file=fout)
                print(file=fout)
print("Time:", time.time() - _ts)

