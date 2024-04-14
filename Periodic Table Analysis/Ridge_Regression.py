import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
def calculate_RSOS(y_true, y_pred):
    residuals = y_true - y_pred
    RSOS = np.sum(residuals**2)
    return RSOS
df = pd.read_csv("Modern_Periodic_Table.csv")
df.dropna(subset=['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                  'Specific Heat', 'Thermal Conductivity', 'Electronegativity'], inplace=True)
X = df[['Atomic Number', 'Density', 'Melting Point', 'Boiling Point',
        'Specific Heat', 'Thermal Conductivity', 'Electronegativity']]
y = df['Atomic Mass']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    plt.scatter(df[feature], y, color='blue', alpha=0.6)
    plt.title(f'{feature} vs Atomic Mass')
    plt.xlabel(feature)
    plt.ylabel('Atomic Mass')
plt.tight_layout()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
alpha = 0.1
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean squared error =', mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R² score) =', r2_score(y_test, y_pred))
RSOS = calculate_RSOS(y_test, y_pred)
print('Residual Sum of Squares (RSOS) =', RSOS)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Atomic Mass')
plt.ylabel('Predicted Atomic Mass')
plt.title('Ridge Regression: Actual vs Predicted Atomic Mass')
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.coef_)), model.coef_, marker='o', linestyle='-', color='blue')
plt.xticks(range(len(X.columns)), X.columns, rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients')
plt.grid(True)
plt.show()
