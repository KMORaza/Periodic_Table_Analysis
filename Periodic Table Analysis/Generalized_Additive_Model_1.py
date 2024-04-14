from pygam import GAM, s
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
target_variable = 'Atomic Mass'
selected_features = ['Atomic Number', 'Density', 'Melting Point', 'Boiling Point']
data = data[selected_features + [target_variable]]
data.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data[target_variable], test_size=0.2, random_state=42)
gam = GAM(s(0) + s(1) + s(2) + s(3))
gam.fit(X_train, y_train)
predictions = gam.predict(X_test)
explained_variance = 1 - np.var(y_test - predictions) / np.var(y_test)
print("R-squared:", explained_variance)
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 5))
for i, feature in enumerate(selected_features):
    feature_values = np.linspace(np.min(data[feature]), np.max(data[feature]), 100)
    X_grid = X_test.copy()
    X_grid[feature] = np.mean(X_test[feature])
    X_grid = pd.concat([X_grid]*100, ignore_index=True)
    X_grid[feature] = np.tile(feature_values, len(X_test))
    predictions_grid = gam.predict(X_grid)
    predictions_grid = np.reshape(predictions_grid, (len(X_test), -1))
    axs[i].plot(feature_values, np.mean(predictions_grid, axis=0), color='red')
    axs[i].scatter(X_test[feature], predictions, color='blue')
    axs[i].set_title("Partial Dependence Plot for {}".format(feature))
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("Atomic Mass")
plt.tight_layout()
plt.show()
