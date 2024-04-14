import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
data = pd.read_csv("Modern_Periodic_Table.csv")
melting_points = data['Melting Point']
boiling_points = data['Boiling Point']
data.dropna(subset=['Melting Point'], inplace=True)
melting_points_np = data['Melting Point'].values
boiling_points_np = data['Boiling Point'].values
imputer = SimpleImputer(strategy='mean')
boiling_points_np_imputed = imputer.fit_transform(boiling_points_np.reshape(-1, 1))
scaler = StandardScaler()
boiling_points_np_scaled = scaler.fit_transform(boiling_points_np_imputed)
ridge_regression = Ridge(alpha=1.0)
ridge_regression.fit(boiling_points_np_scaled, melting_points_np)
coefficients = ridge_regression.coef_
intercept = ridge_regression.intercept_
predicted_melting_points = ridge_regression.predict(boiling_points_np_scaled)
r_squared = r2_score(melting_points_np, predicted_melting_points)
plt.figure(figsize=(10, 6))
plt.scatter(boiling_points_np_scaled, melting_points_np, label='Original Data')
plt.plot(boiling_points_np_scaled, predicted_melting_points, color='red', label='Predicted Data')
plt.xlabel('Boiling Point (Scaled)')
plt.ylabel('Melting Point')
plt.title('Functional Regression: Melting Point vs. Boiling Point')
plt.legend()
plt.show()
print("Coefficients =", coefficients)
print("Intercept =", intercept)
print("R-squared =", r_squared)
