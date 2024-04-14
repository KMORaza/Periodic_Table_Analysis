import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = ['Atomic Number', 'State', 'Boiling Point']
data = data[selected_columns]
data.dropna(inplace=True)
X_level1 = data[['Atomic Number']]
y = data['Boiling Point']
X_level1 = sm.add_constant(X_level1)
model_level1 = sm.OLS(y, X_level1).fit()
print("📝 LEVEL-1 MODEL SUMMARY:")
print(model_level1.summary())
coefficients_level1 = pd.DataFrame({
    'Coefficient': model_level1.params,
    'Standard Error': model_level1.bse,
    '95% CI Lower': model_level1.conf_int()[0],
    '95% CI Upper': model_level1.conf_int()[1]
})
print(tabulate(coefficients_level1, headers='keys', tablefmt='psql'))
plt.figure(figsize=(8, 6))
plt.scatter(X_level1['Atomic Number'], y, label='Data')
plt.plot(X_level1['Atomic Number'], model_level1.predict(X_level1), color='red', label='Level 1 Regression')
plt.xlabel('Atomic Number')
plt.ylabel('Boiling Point')
plt.title('Level-1')
plt.legend()
plt.show()
print("\nLevel-1 Model: Boiling Point = {} + {} * Atomic Number + ε".format(model_level1.params['const'], model_level1.params['Atomic Number']))
print("\nCoefficients:")
print(model_level1.params)
print("\nStandard Errors:")
print(model_level1.bse)
print("\nConfidence Intervals:")
print(model_level1.conf_int())
print("\nR-squared =",model_level1.rsquared)
X_level2 = pd.get_dummies(data['State'], drop_first=True)
groups = data['State']
model_level2 = sm.MixedLM(endog=y, exog=X_level2, groups=groups).fit()
print("\n📝 LEVEL-2 MODEL SUMMARY:")
print(model_level2.summary())
coefficients_level2 = pd.DataFrame({
    'Coefficient': model_level2.params,
    'Standard Error': model_level2.bse,
    '95% CI Lower': model_level2.conf_int()[0],
    '95% CI Upper': model_level2.conf_int()[1]
})
print(tabulate(coefficients_level2, headers='keys', tablefmt='psql'))
plt.figure(figsize=(8, 6))
plt.scatter(X_level2.index, y, label='Data')
plt.scatter(X_level2.index, model_level2.fittedvalues, color='red', label='Level 2 Predictions')
plt.xlabel('Observation Index')
plt.ylabel('Boiling Point')
plt.title('Level-2')
plt.legend()
plt.show()
print("\nLevel-2 Model: Boiling Point = {} + ε".format(model_level2.params[0]))
print("\nRandom Effects (Group-Specific Intercept):")
print(model_level2.random_effects)
print("\nStandard Errors of Random Effects:")
print(model_level2.bse_fe)
print("\nVariance of Random Effects:")
print(model_level2.cov_re)
