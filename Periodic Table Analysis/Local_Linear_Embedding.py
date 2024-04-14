import pandas as pd
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
features = data[['Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity', 'Thermal Expansion']]
imputer = SimpleImputer(strategy='mean')
imputed_features = imputer.fit_transform(features)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
embedded_data = lle.fit_transform(scaled_features)
plt.figure(figsize=(8, 6))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=data['Atomic Number'], cmap='viridis', edgecolor='k', alpha=0.7)
plt.title('Local Linear Embedding of Periodic Table Data')
plt.xlabel('Embedded Dimension 1')
plt.ylabel('Embedded Dimension 2')
plt.colorbar(label='Atomic Number')
plt.grid(True)
plt.show()
