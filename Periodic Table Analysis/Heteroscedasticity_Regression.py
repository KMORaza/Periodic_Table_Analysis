import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Modern_Periodic_Table.csv')
data.dropna(subset=['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point'], inplace=True)
X = data[['Atomic Mass', 'Melting Point', 'Boiling Point']]
Y = data['Density']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
residuals = model.resid
residual_var = np.var(residuals)
weights = 1 / np.sqrt(np.abs(residuals))
wls_model = sm.WLS(Y, X, weights=weights).fit()
_, bp_pvalue, _, _ = het_breuschpagan(wls_model.resid, wls_model.model.exog)
wls_model_summary = wls_model.summary(title="")
print(wls_model_summary)
print("\nBreusch-Pagan test p-value =", bp_pvalue)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=wls_model.fittedvalues, y=wls_model.resid)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Observed vs Predicted Values")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
plt.figure(figsize=(10, 6))
sns.residplot(x=wls_model.fittedvalues, y=wls_model.resid)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(wls_model.resid, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()
print("\nCoefficients:")
print(wls_model.params)
print("\nStandard Errors:")
print(wls_model.bse)
print("\nT-values:")
print(wls_model.tvalues)
print("\nP-values:")
print(wls_model.pvalues)
print("\nConfidence Intervals:")
print(wls_model.conf_int())
print("\nR-squared =", wls_model.rsquared)
print("\nAdjusted R-squared =", wls_model.rsquared_adj)
