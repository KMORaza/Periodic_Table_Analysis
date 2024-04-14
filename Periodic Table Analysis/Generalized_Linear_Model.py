import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = ['Atomic Mass', 'Atomic Number', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion']
filtered_data = data[selected_columns].dropna()
formula = 'Q("Atomic Mass") ~ Q("Atomic Number") + Density + Q("Melting Point") + Q("Boiling Point") + Q("Heat of Fusion")'
glm_model = smf.glm(formula=formula, data=filtered_data, family=sm.families.Gaussian()).fit()
print(glm_model.summary())
plt.figure(figsize=(8, 6))
plt.scatter(glm_model.fittedvalues, filtered_data['Atomic Mass'], alpha=0.5)
plt.plot(filtered_data['Atomic Mass'], filtered_data['Atomic Mass'], color='red', linestyle='--')
plt.title('Observed vs. Predicted Atomic Mass')
plt.xlabel('Predicted Atomic Mass')
plt.ylabel('Observed Atomic Mass')
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(glm_model.fittedvalues, glm_model.resid_deviance, alpha=0.5)
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(glm_model.resid_deviance, bins=20, edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
