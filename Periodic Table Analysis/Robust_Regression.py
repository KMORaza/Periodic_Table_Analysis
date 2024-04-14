import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
data = pd.read_csv("Modern_Periodic_Table.csv")
features = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
            'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity']
X = data[features]
y = data['Electronegativity']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)
missing_target_indices = y.isnull()
X_imputed = X_imputed[~missing_target_indices]
y = y.dropna()
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
huber = HuberRegressor()
huber.fit(X_train, y_train)
y_pred = huber.predict(X_test)
print("Coefficients:", huber.coef_)
print("Intercept:", huber.intercept_)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Electronegativity')
plt.ylabel('Predicted Electronegativity')
plt.title('Predicted vs Actual Electronegativity')
plt.grid(True)
plt.show()
