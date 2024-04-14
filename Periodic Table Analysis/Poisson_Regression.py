import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = ['Atomic Number', 'Density', 'Melting Point', 'Boiling Point',
                    'Heat of Fusion', 'Heat of Vaporization', 'Thermal Conductivity', 'Atomic Radius']
data_selected = data[selected_columns]
data_selected.dropna(inplace=True)
X = data_selected.drop(['Atomic Number'], axis=1)
X = sm.add_constant(X)
y = data_selected['Atomic Number']
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(poisson_model.summary())
predicted = poisson_model.predict(X)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=predicted)
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Values')
plt.show()
residuals = y - predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=residuals)
plt.xlabel('Observed')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
