import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Modern_Periodic_Table.csv")
df.dropna(subset=['Atomic Number', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Atomic Mass',
                  'Specific Heat', 'Thermal Conductivity', 'Electronegativity', 'Atomic Radius', 'Molar Volume',
                  'Thermal Expansion'], inplace=True)
X = df[['Atomic Number', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Specific Heat',
        'Thermal Conductivity', 'Electronegativity', 'Atomic Radius', 'Molar Volume', 'Thermal Expansion']]
y = df['Atomic Mass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
degree = 3
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
print("Coefficients:")
coefficients = model.coef_
for i, coef in enumerate(coefficients):
    print(f"Feature {i}: {coef}")
print(f"Intercept: {model.intercept_}")
plt.figure(figsize=(12, 8))
sns.heatmap(coefficients.reshape(1, -1), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Coefficients Matrix")
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.xticks(ticks=np.arange(len(poly.get_feature_names_out(X.columns))), labels=poly.get_feature_names_out(X.columns), rotation=45, ha='right')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()
fig, axes = plt.subplots(4, 3, figsize=(20, 15))
for i, ax in enumerate(axes.flat):
    if i < X.shape[1]:
        ax.scatter(X.iloc[:, i], y, alpha=0.5)
        ax.set_xlabel(X.columns[i])
        ax.set_ylabel('Atomic Mass')
plt.tight_layout()
plt.show()
for i, feature in enumerate(X.columns):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[feature], y, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Atomic Mass')
    plt.title(f'{feature} vs. Atomic Mass')
    plt.grid(True)
    plt.show()
print('Mean squared error =', mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R² score) =', r2_score(y_test, y_pred))
