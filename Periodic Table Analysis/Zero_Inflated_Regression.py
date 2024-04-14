import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
selected_columns = ['Element Name', 'Density', 'Melting Point', 'Boiling Point', 'Atomic Mass', 'Atomic Number'] # Add more columns if needed
data = data[selected_columns]
data.dropna(inplace=True)
X = data[['Melting Point', 'Boiling Point', 'Atomic Mass', 'Atomic Number']]
y = data['Density']
X = sm.add_constant(X)
model = ZeroInflatedPoisson(y, X).fit()
print(model.summary())
print("\n")
print("Log-likelihood =", model.llf)
print("Akaike Information Criterion =", model.aic)
print("Bayesian Information Criterion =", model.bic)
print("Inflation parameter =", model.params[-1])
print("\nParameter estimates for zero-inflation component:")
print(model.params[:-1])
print("\nLikelihood ratio test (LR test) =", model.llr)
print("\n")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
predicted = model.predict(X)
residuals = model.resid
ax[0].scatter(predicted, residuals)
ax[0].axhline(y=0, color='r', linestyle='-')
ax[0].set_xlabel("Fitted values")
ax[0].set_ylabel("Residuals")
ax[0].set_title('Residuals vs Fitted')
sm.qqplot(model.resid_pearson, line='45', ax=ax[1])
ax[1].set_title('QQ Plot of Residuals')
plt.show()
