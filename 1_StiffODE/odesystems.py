import numpy as np


class ODESystem:
    def __init__(self, msg, f, u0, t0, T):
        self.msg = msg
        self.f = f
        self.u0 = u0
        self.t0 = t0
        self.T = T


class LinearSystem(ODESystem):
    def __init__(self, a, b, **kwargs):
        super().__init__(
            'Линейная однородная система',
            lambda t, u: np.asarray([
                a * u[0],
                u[0] + a * u[1],
                b * u[2],
                u[2] + b * u[3],
                2 * u[3] + b * u[4],
                3 * u[4] + b * u[5],
            ]), **kwargs)


class AutonomousSystem(ODESystem):
    def __init__(self, a, b, c, **kwargs):
        super().__init__(
            'Нелинейная автономная система',
            lambda t, u: np.asarray([
                a / b * u[1] + c * u[0] * (np.sqrt((u[0] / a) ** 2 + (u[1] / b) ** 2) - 1),
                -b / a * u[0] + c * u[1] * (np.sqrt((u[0] / a) ** 2 + (u[1] / b) ** 2) - 1),
            ]), **kwargs)


class ConstEigenSystem(ODESystem):
    def __init__(self, a, b, c, **kwargs):
        super().__init__(
            'Нелинейная система с постоянными собственными числами',
            lambda t, u: np.asarray([
                (a - b * np.cos(2 * c * t)) * u[0] + (b * np.sin(2 * c * t) + c) * u[1],
                (b * np.sin(2 * c * t) - c) * u[0] + (a + b * np.cos(2 * c * t)) * u[1],
            ]), **kwargs)


class NonLinearSystem(ODESystem):
    def __init__(self, a, b, c, **kwargs):
        super().__init__(
            'Нелинейная система',
            lambda t, u: np.asarray([
                a ** 3 / (3 * b ** 3) * (u[1] ** 3) / (u[0] ** 2) + c * u[0] * (np.sqrt((u[0] / a) ** 6 + (u[1] / b) ** 6) - 1),
                -b ** 3 / (3 * a ** 3) * (u[0] ** 3) / (u[1] ** 2) + c * u[1] * (np.sqrt((u[0] / a) ** 6 + (u[1] / b) ** 6) - 1),
            ]), **kwargs)


class SimpleSystem(ODESystem):
    def __init__(self, a, b, **kwargs):
        super().__init__(
            'Простая нелинейная система',
            lambda t, u: np.asarray([
                a * u[1],
                np.sin(u[0]) + b * u[1]
            ]), **kwargs)
