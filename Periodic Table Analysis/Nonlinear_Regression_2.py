import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
data = pd.read_csv('Modern_Periodic_Table.csv')
data = data.dropna(subset=['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                           'Specific Heat', 'Thermal Conductivity', 'Molar Volume', 'Atomic Radius'])
x1_data = data['Atomic Number']
x2_data = data['Density']
x3_data = data['Melting Point']
x4_data = data['Boiling Point']
x5_data = data['Specific Heat']
x6_data = data['Thermal Conductivity']
x7_data = data['Molar Volume']
x8_data = data['Atomic Radius']
y_data = data['Atomic Mass']
def nonlinear_model(x, a, b, c, d, e, f, g, h, i):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    return a * np.exp(b * x1) + c * x2 + d * x3 + e * x4 + f * x5 + g * x6 + h * x7 + i * x8
initial_guess = (1, 1, 1, 1, 1, 1, 1, 1, 1)
params, covariance = curve_fit(nonlinear_model, (x1_data, x2_data, x3_data, x4_data, x5_data, x6_data, x7_data, x8_data), y_data, p0=initial_guess)
a, b, c, d, e, f, g, h, i = params
y_pred = nonlinear_model((x1_data, x2_data, x3_data, x4_data, x5_data, x6_data, x7_data, x8_data), a, b, c, d, e, f, g, h, i)
r_squared = r2_score(y_data, y_pred)
print("Parameters:")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)
print("f =", f)
print("g =", g)
print("h =", h)
print("i =", i)
print("R-squared value:", r_squared)
plt.figure(figsize=(10, 6))
plt.scatter(y_data, y_pred, color='blue', label='Original vs Predicted')
plt.plot([min(y_data), max(y_data)], [min(y_data), max(y_data)], '--', color='red', label='Perfect prediction')
plt.xlabel('Original Atomic Mass')
plt.ylabel('Predicted Atomic Mass')
plt.title('Original vs Predicted Atomic Mass (Nonlinear Regression)')
plt.legend()
plt.grid(True)
plt.show()
residuals = y_data - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='green')
plt.xlabel('Predicted Atomic Mass')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='red', linestyle='--')
plt.grid(True)
plt.show()
