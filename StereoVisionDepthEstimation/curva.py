import numpy as np
from scipy.interpolate import interp1d

# Definir los valores de muestra
x_data = [100, 200, 300, 400, 500]
y_data = [183, 270, 338, 386, 436]

# Crear una función de interpolación lineal
interp_func = interp1d(x_data, y_data, kind='cubic')

# Utilizar la función de interpolación para ajustar los valores a 100 cm
y_interp = interp_func(250)

# Imprimir los valores ajustados
print("Valor ajustado a 100 cm:", y_interp)