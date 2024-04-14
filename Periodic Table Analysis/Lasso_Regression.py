import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Modern_Periodic_Table.csv")
relevant_columns = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                    'Specific Heat', 'Thermal Conductivity', 'Electronegativity', 'Neel Point',
                    'Electrical Conductivity', 'Thermal Expansion']
df.dropna(subset=relevant_columns, inplace=True)
X = df[['Atomic Number', 'Density', 'Melting Point', 'Boiling Point',
        'Specific Heat', 'Thermal Conductivity', 'Electronegativity', 'Neel Point',
        'Electrical Conductivity', 'Thermal Expansion']]
y = df['Atomic Mass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
alpha = 0.1
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error = {mse}")
print(f"R-squared = {r2}")
plt.figure(figsize=(8, 4))
coef = pd.Series(lasso.coef_, index=X.columns)
imp_coef = pd.concat([coef.sort_values().head(10),
                      coef.sort_values().tail(10)])
imp_coef.plot(kind='barh')
plt.title("Feature Importance (Lasso)")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.show()
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    axes[i].scatter(X[col], y, alpha=0.5)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Atomic Mass')
plt.tight_layout()
plt.show()
