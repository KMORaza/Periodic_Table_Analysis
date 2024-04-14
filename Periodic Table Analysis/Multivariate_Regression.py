import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
df = pd.read_csv("Modern_Periodic_Table.csv")
X_cols = ['Atomic Number', 'Atomic Mass', 'Heat of Fusion', 'Heat of Vaporization', 'Heat of Combustion',
          'Heat of Combustion', 'Specific Heat', 'Thermal Conductivity', 'Bulk Modulus', 'Young\'s Modulus']
y_cols = ['Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization',
          'Heat of Combustion', 'Specific Heat', 'Thermal Conductivity', 'Bulk Modulus', 'Young\'s Modulus']
df.dropna(subset=X_cols + y_cols, inplace=True)
X = df[X_cols]
y = df[y_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
model = LinearRegression()
model.fit(X_train_imputed, y_train)
y_pred = model.predict(X_test_imputed)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficients:")
for i, col in enumerate(X_cols):
    print(f"{col}: {model.coef_[:, i]}")
print("Intercepts:")
for i, col in enumerate(y_cols):
    print(f"{col}: {model.intercept_[i]}")
print("\nPredicted vs. Actual:")
for i in range(min(5, len(y_test))):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.values[i]}")
