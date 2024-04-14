import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Modern_Periodic_Table.csv')
variables_of_interest = ['Density', 'Melting Point', 'Boiling Point', 'Atomic Mass', 'Electronegativity',
                         'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity',
                         'Atomic Radius', 'Covalent Radius', 'van der Waals Radius']
data.dropna(subset=variables_of_interest, inplace=True)
X = data[['Melting Point', 'Boiling Point', 'Atomic Mass', 'Electronegativity',
          'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity',
          'Atomic Radius', 'Covalent Radius', 'van der Waals Radius']]
y = data['Density']
censoring_threshold = 0
censored = (y < censoring_threshold)
y_censored = y.copy()
y_censored[y_censored < censoring_threshold] = censoring_threshold
X = sm.add_constant(X)
tobit_model = sm.GLM(y_censored, X, family=sm.families.Tweedie(var_power=0))
tobit_results = tobit_model.fit()
y_pred = tobit_results.predict(X)
y_pred[censored] = censoring_threshold
print(tobit_results.summary())
residuals = tobit_results.resid_response
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
std_residuals = tobit_results.resid_pearson
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, std_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Standardized Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Standardized Residuals')
plt.grid(True)
plt.show()
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()
sqrt_abs_std_residuals = np.sqrt(np.abs(std_residuals))
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, sqrt_abs_std_residuals)
plt.title('Scale-Location Plot')
plt.xlabel('Fitted Values')
plt.ylabel('sqrt(|Standardized Residuals|)')
plt.grid(True)
plt.show()
