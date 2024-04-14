import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
X = data[['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point']]
y_electronegativity = data['Electronegativity']
y_density = data['Density']
y_melting_point = data['Melting Point']
y_electronegativity_imputed = SimpleImputer(strategy='mean').fit_transform(y_electronegativity.values.reshape(-1, 1)).ravel()
y_density_imputed = SimpleImputer(strategy='mean').fit_transform(y_density.values.reshape(-1, 1)).ravel()
y_melting_point_imputed = SimpleImputer(strategy='mean').fit_transform(y_melting_point.values.reshape(-1, 1)).ravel()
X_imputed = SimpleImputer(strategy='mean').fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
bayesian_reg_electronegativity = BayesianRidge()
bayesian_reg_electronegativity.fit(X_scaled, y_electronegativity_imputed)
bayesian_reg_density = BayesianRidge()
bayesian_reg_density.fit(X_scaled, y_density_imputed)
bayesian_reg_melting_point = BayesianRidge()
bayesian_reg_melting_point.fit(X_scaled, y_melting_point_imputed)
X_new = np.array([
    [80, 200, 10, 1000, 2000],
    [90, 210, 12, 1100, 2100],
    [100, 220, 14, 1200, 2200]
])
X_new_imputed = SimpleImputer(strategy='mean').fit_transform(X_new)
X_new_scaled = scaler.transform(X_new_imputed)
electronegativity_pred, electronegativity_std = bayesian_reg_electronegativity.predict(X_new_scaled, return_std=True)
density_pred, density_std = bayesian_reg_density.predict(X_new_scaled, return_std=True)
melting_point_pred, melting_point_std = bayesian_reg_melting_point.predict(X_new_scaled, return_std=True)
print("Predicted Results:")
for i in range(len(X_new)):
    print(f"Data Point {i+1}:")
    print(f"  Electronegativity: Mean = {electronegativity_pred[i]}, Std = {electronegativity_std[i]}")
    print(f"  Density: Mean = {density_pred[i]}, Std = {density_std[i]}")
    print(f"  Melting Point: Mean = {melting_point_pred[i]}, Std = {melting_point_std[i]}")
