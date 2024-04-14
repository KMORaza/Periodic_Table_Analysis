import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
data = pd.read_csv('Modern_Periodic_Table.csv')
response_variable = 'Density'
predictor_variables = ['Atomic Mass', 'Melting Point', 'Boiling Point', 'Specific Heat',
                       'Atomic Number', 'Heat of Fusion', 'Heat of Vaporization',
                       'Brinell Hardness', 'Electronegativity']
data = data.dropna(subset=[response_variable] + predictor_variables)
data['intercept'] = 1
X = data[['intercept'] + predictor_variables]
y = data[response_variable]
negbin_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
print(negbin_model.summary())
print("\nConfidence Intervals:")
print(negbin_model.conf_int())
print("\nDispersion Parameter (α) =", negbin_model.scale)
plt.figure(figsize=(10, 6))
plt.scatter(negbin_model.fittedvalues, y, c='b', alpha=0.6)
plt.plot(negbin_model.fittedvalues, negbin_model.fittedvalues, c='r')
plt.title('Observed vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Observed Values')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(negbin_model.fittedvalues, negbin_model.resid_response, c='b', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
