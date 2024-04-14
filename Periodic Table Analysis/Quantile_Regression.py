import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
X = data[['Atomic Mass']].values
y = data['Density'].values
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]
X = X[~np.isinf(y)]
y = y[~np.isinf(y)]
quantiles = [0.25, 0.5, 0.75]
models = []
for quantile in quantiles:
    gam = LinearGAM(s(0))
    gam.fit(X, y)
    models.append(gam)
for i, quantile in enumerate(quantiles):
    print(f"Quantile Regression Results for quantile {quantile}:")
    print(f"Coefficients: {models[i].coef_}")
quantile = 0.5
predicted_values = models[quantiles.index(quantile)].predict(X)
print("Predicted values at 0.5 quantile:")
print(predicted_values)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data', alpha=0.3)
for i, quantile in enumerate(quantiles):
    XX = np.linspace(X.min(), X.max(), 100)
    plt.plot(XX, models[i].predict(XX), label=f'Quantile {quantile}')
plt.title('Quantile Regression')
plt.xlabel('Atomic Mass')
plt.ylabel('Density')
plt.legend()
plt.show()
