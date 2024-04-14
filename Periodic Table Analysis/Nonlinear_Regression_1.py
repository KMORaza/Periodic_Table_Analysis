import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
data = pd.read_csv('Modern_Periodic_Table.csv')
data = data.dropna(subset=['Atomic Number', 'Atomic Mass'])
x_data = data['Atomic Number']
y_data = data['Atomic Mass']
def nonlinear_model(x, a, b):
    return a * np.exp(b * x)
params, covariance = curve_fit(nonlinear_model, x_data, y_data)
a, b = params
y_pred = nonlinear_model(x_data, a, b)
r_squared = r2_score(y_data, y_pred)
print("Parameters:")
print("a =", a)
print("b =", b)
print("R-squared value:", r_squared)
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, y_pred, color='red', label='Fitted Curve')
plt.xlabel('Atomic Number')
plt.ylabel('Atomic Mass')
plt.title('Nonlinear Regression')
plt.legend()
plt.show()
