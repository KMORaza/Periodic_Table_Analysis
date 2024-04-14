import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Modern_Periodic_Table.csv')
features = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point']
target = 'Specific Heat'
data.dropna(subset=features + [target], inplace=True)
X = data[features]
y = data[target]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
y_pred_train = svr_model.predict(X_train_scaled)
y_pred_test = svr_model.predict(X_test_scaled)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Actual vs Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='red', label='Ideal')
plt.title('Actual vs Predicted (Training Set)')
plt.xlabel('Actual Specific Heat')
plt.ylabel('Predicted Specific Heat')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='green', label='Actual vs Predicted (Testing)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Ideal')
plt.title('Actual vs Predicted (Testing Set)')
plt.xlabel('Actual Specific Heat')
plt.ylabel('Predicted Specific Heat')
plt.legend()
plt.grid(True)
plt.show()
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred_test, y=residuals, color='orange')
plt.title('Residual Plot')
plt.xlabel('Predicted Specific Heat')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
