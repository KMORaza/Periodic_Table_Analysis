import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
params = {'alpha': [0.001, 0.01, 0.1, 1, 10],
          'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
enet = ElasticNet(max_iter=10000)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(enet, params, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_enet = grid_search.best_estimator_
best_enet.fit(X_train, y_train)
train_score = mean_squared_error(y_train, best_enet.predict(X_train))
test_score = mean_squared_error(y_test, best_enet.predict(X_test))
print("Best hyperparameters:", grid_search.best_params_)
print("Train MSE:", train_score)
print("Test MSE:", test_score)
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(X.shape[1]), best_enet.coef_)
plt.title("Coefficients in the ElasticNet Model")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.show()
