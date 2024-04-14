import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = pd.read_csv("Modern_Periodic_Table.csv")
features = ['Atomic Number', 'Period', 'Graph Period']
target = 'Half-Life'
X = data[features].values
y = data[target].values
y = np.where(np.isinf(y), np.nan, y)
y = np.where(y > 1e6, np.nan, y)
nan_indices = np.isnan(y)
X = X[~nan_indices]
y = y[~nan_indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = np.log(y_train)
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp_model.fit(X_train, y_train)
y_pred_log, y_std_log = gp_model.predict(X_test, return_std=True)
y_pred = np.exp(y_pred_log)
y_std = np.exp(y_std_log)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error =", mse)
print("\nKernel Parameters:")
print("Constant Kernel Parameter =", gp_model.kernel_.k1.get_params()['constant_value'])
print("RBF Kernel Parameter (Length Scale) =", gp_model.kernel_.k2.get_params()['length_scale'])
print("\nModel Coefficients:")
print("Coefficients =", gp_model.alpha_)
