import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from tabulate import tabulate 
data = pd.read_csv('Modern_Periodic_Table.csv')
X = data[['Density', 'Melting Point', 'Boiling Point', 'Thermal Conductivity', 'Specific Heat', 'Thermal Expansion']]
Y = data['Atomic Mass']
data.dropna(subset=['Density', 'Melting Point', 'Boiling Point', 'Thermal Conductivity', 'Specific Heat', 'Thermal Expansion', 'Atomic Mass'], inplace=True)
iv_model = IV2SLS(dependent=Y, exog=X, endog=None, instruments=None)
iv_results = iv_model.fit()
print(iv_results.summary)
print("\nCoefficient Estimates:")
print(iv_results.params)
print("\nStandard Errors:")
print(iv_results.std_errors)
print("\nT-values:")
print(iv_results.tstats)
print("\nP-values:")
print(iv_results.pvalues)
print("\nR-squared:")
print(iv_results.rsquared)
print("\nF-statistic:")
print(iv_results.f_statistic.stat)
plt.figure(figsize=(8, 6))
plt.scatter(iv_results.predict(), iv_results.resids)
plt.title("Actual vs. Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()
coef_estimates = iv_results.params.rename('Parameter')
std_errors = iv_results.std_errors.rename('Std. Error')
t_values = iv_results.tstats.rename('T-stat')
p_values = iv_results.pvalues.rename('P-value')
results_df = pd.concat([coef_estimates, std_errors, t_values, p_values], axis=1)
print(tabulate(results_df, headers='keys', tablefmt='grid'))
