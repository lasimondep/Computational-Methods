from numpy import array

# Коэффициенты для метода Адамса
k_Adams = 4
b_Adams = array((251, 646, -264, 106, -19)) / 720

# Шаг для численного дифференцирования
h_jac = 5e-4

# Коэффициенты для метода Розенброка
a_CROS1 = (1 + 1j) / 2
b_CROS1 = 1
c_CROS1 = 0.5
