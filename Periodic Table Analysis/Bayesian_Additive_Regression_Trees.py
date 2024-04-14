import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
data.dropna(subset=['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Specific Heat'], inplace=True)
X = data[['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
          'Molar Volume', 'Atomic Radius', 'Electronegativity']]
y = data['Specific Heat']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(X_train_scaled, y_train)
predictions = bayesian_ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)
print("Mean Squared Error =", mse)
print("R-squared Score =", r_squared)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
residuals = y_test - predictions
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()
print("Model Coefficients:")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {bayesian_ridge.coef_[i]}")
print(" ")
print("Intercept =", bayesian_ridge.intercept_)
print("Estimated Precision of the noise =", bayesian_ridge.lambda_)
print("Estimated Variance =", bayesian_ridge.alpha_)
print(" ")
coefficients = [(feature, coef) for feature, coef in zip(X.columns, bayesian_ridge.coef_)]
highlight_threshold = 0.001
rows = []
for feature, coef in coefficients:
    coef_str = f"{coef:.6f}" if abs(coef) > highlight_threshold else f"{coef:.6f}"
    rows.append([feature, coef_str])
table_str = tabulate(rows, headers=["Feature", "Coefficient"], tablefmt="fancy_grid", numalign="center", stralign="center")
print(table_str)
