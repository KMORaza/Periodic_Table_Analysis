import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Modern_Periodic_Table.csv")
df.dropna(subset=['Atomic Number', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Atomic Mass',
                  'Specific Heat', 'Thermal Conductivity', 'Electronegativity'], inplace=True)
X = df[['Atomic Number', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion',
        'Specific Heat', 'Thermal Conductivity', 'Electronegativity']]
y = df['Atomic Mass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)
plt.show()
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    if i < X.shape[1]:
        ax.scatter(X.iloc[:, i], y, alpha=0.5)
        ax.set_xlabel(X.columns[i])
        ax.set_ylabel('Atomic Mass')
plt.tight_layout()
plt.show()
print('Mean squared error =', mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R² score) =', r2_score(y_test, y_pred))
