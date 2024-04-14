import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
embedding_features = ['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                      'Specific Heat', 'Thermal Conductivity', 'Electronegativity']
data_subset = data.dropna(subset=embedding_features)
X = data_subset[embedding_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
spectral_embedding = SpectralEmbedding(n_components=2, affinity='nearest_neighbors')
X_embedding = spectral_embedding.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], s=50, c='blue', alpha=0.5)
plt.title('Spectral Embedding of Periodic Table Data')
plt.xlabel('Embedded Dimension 1')
plt.ylabel('Embedded Dimension 2')
plt.grid(True)
plt.show()
