import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
data = pd.read_csv("Modern_Periodic_Table.csv")
data_numeric = data.select_dtypes(include=[np.number])
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
data_numeric = data_numeric.fillna(data_numeric.mean())
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)
cca = KernelPCA(n_components=2, kernel='rbf')
cca_pipeline = make_pipeline(scaler, cca)
transformed_data = cca_pipeline.fit_transform(data_numeric)
print(transformed_data)
plt.figure(figsize=(10, 6))
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
plt.title('Curvilinear Component Analysis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()
