# Multidimensional Scaling (MDS)
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = [
    'Atomic Mass', 
    'Density', 
    'Melting Point', 
    'Boiling Point', 
    'Specific Heat', 
    'Thermal Conductivity',
    'Electronegativity',
    'Atomic Radius',
    'Covalent Radius',
    'van der Waals Radius'
]
data = data.dropna(subset=selected_columns)
X = data[selected_columns].values
mds = MDS(n_components=2, dissimilarity='euclidean')
X_transformed = mds.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='blue', alpha=0.5)
for i, symbol in enumerate(data['Element Symbol']):
    plt.annotate(symbol, (X_transformed[i, 0], X_transformed[i, 1]), fontsize=8)
plt.xlabel('MDS Component 1')
plt.ylabel('MDS Component 2')
plt.title('Multidimensional Scaling on Periodic Table Data')
plt.grid(True)
plt.show()
