import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
columns_for_clustering = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity', 'Molar Volume', 'Brinell Hardness', 'Mohs Hardness', 'Bulk Modulus', 'Shear Modulus', 'Young\'s Modulus', 'Refractive Index', 'Speed of Sound', 'Valency', 'Electronegativity', 'Electron Affinity', 'Autoignition Point', 'Flash Point', 'Atomic Radius', 'Covalent Radius', 'van der Waals Radius', 'Mass Magnetic Susceptibility', 'Molar Magnetic Susceptibility', 'Volume Magnetic Susceptibility']
X = data[columns_for_clustering]
X.fillna(X.mean(), inplace=True)
Z = linkage(X, 'ward')
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Elements')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., labels=data['Element Symbol'].values)
plt.show()
max_d = 1200
clusters = fcluster(Z, max_d, criterion='distance')
cluster_details = pd.DataFrame({'Element Symbol': data['Element Symbol'], 'Cluster': clusters})
print(tabulate(cluster_details, headers='keys', tablefmt='grid'))
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.5)
plt.title('Hierarchical Clustering - Scatter Plot')
plt.xlabel(columns_for_clustering[0])
plt.ylabel(columns_for_clustering[1])
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
num_clusters = cluster_details['Cluster'].nunique()
print(f"Number of clusters = {num_clusters}\n")
print("Elements in each cluster:")
cluster_elements = cluster_details.groupby('Cluster')['Element Symbol'].apply(list).reset_index(name='Elements')
print(tabulate(cluster_elements, headers='keys', tablefmt='grid'), "\n")
cluster_stats = cluster_details.groupby('Cluster').size().reset_index(name='Count')
cluster_stats['Percentage'] = cluster_stats['Count'] / len(cluster_details) * 100
print("Statistics about the clusters:")
print(tabulate(cluster_stats, headers='keys', tablefmt='grid'))
centroids = X.groupby(clusters).mean()
print("\nCentroids of clusters:")
print(tabulate(centroids, headers='keys', tablefmt='grid'))
