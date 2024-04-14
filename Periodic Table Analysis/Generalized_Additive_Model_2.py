from pygam import GAM, s
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
target_variable = 'Atomic Mass'
selected_features = ['Atomic Number', 'Density', 'Melting Point', 'Boiling Point',
                     'Atomic Radius', 'Electronegativity', 'Specific Heat', 'Thermal Conductivity',
                     'Heat of Fusion', 'Heat of Vaporization', 'Bulk Modulus', 'Young\'s Modulus']
data = data[selected_features + [target_variable]]
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data[target_variable], test_size=0.2, random_state=42)
gam = GAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11))
gam.fit(X_train, y_train)
predictions = gam.predict(X_test)
explained_variance = 1 - np.var(y_test - predictions) / np.var(y_test)
print("R-squared:", explained_variance)
fig, axs = plt.subplots(3, len(selected_features)//3, figsize=(20, 15))
for i, feature in enumerate(selected_features):
    row = i // (len(selected_features)//3)
    col = i % (len(selected_features)//3)
    feature_values = np.linspace(np.min(data[feature]), np.max(data[feature]), 100)
    X_grid = X_test.copy()
    X_grid[feature] = np.mean(X_test[feature])
    X_grid = pd.concat([X_grid]*100, ignore_index=True)
    X_grid[feature] = np.tile(feature_values, len(X_test))
    predictions_grid = gam.predict(X_grid)
    predictions_grid = np.reshape(predictions_grid, (len(X_test), -1))
    axs[row, col].plot(feature_values, np.mean(predictions_grid, axis=0), color='green')
    axs[row, col].scatter(X_test[feature], predictions, color='red')
    axs[row, col].set_title("Partial Dependence Plot for {}".format(feature))
    axs[row, col].set_xlabel(feature)
    axs[row, col].set_ylabel("Atomic Mass")
plt.tight_layout()
plt.show()
