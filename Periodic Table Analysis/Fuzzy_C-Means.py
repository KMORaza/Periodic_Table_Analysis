import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
features = data[['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization',
                 'Specific Heat', 'Thermal Conductivity', 'Thermal Expansion', 'Molar Volume', 'Brinell Hardness',
                 'Mohs Hardness', 'Vickers Hardness', 'Bulk Modulus', 'Shear Modulus', "Young's Modulus",
                 "Poisson's Ratio", 'Refractive Index', 'Speed of Sound', 'Valency', 'Electronegativity',
                 'Electron Affinity', 'Autoignition Point', 'Flash Point', 'Atomic Radius', 'Covalent Radius',
                 'van der Waals Radius', 'Space Group Number', 'Period', 'Group', 'Atomic Number']]
features.fillna(features.mean(), inplace=True)
features_array = features.values
normalized_data = (features_array - features_array.min(axis=0)) / (features_array.max(axis=0) - features_array.min(axis=0))
num_clusters = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(normalized_data.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_centroids = cntr
cluster_membership = np.argmax(u, axis=0)
data['Cluster'] = cluster_membership
plt.figure(figsize=(10, 6))
for i, centroid in enumerate(cluster_centroids):
    plt.plot(centroid, label=f'Cluster {i+1}', marker='o')
plt.title('Cluster Centroids')
plt.xlabel('Feature Index')
plt.ylabel('Feature Value')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_elements = data[data['Cluster'] == i]
    plt.scatter(cluster_elements.index, cluster_elements['Cluster'], label=f'Cluster {i+1}', alpha=0.5)
plt.title('Cluster Membership for Each Element')
plt.xlabel('Element Index')
plt.ylabel('Cluster')
plt.legend()
plt.grid(True)
plt.show()
print("Cluster centroids:")
for i, centroid in enumerate(cluster_centroids):
    print(f"Cluster {i+1} = {centroid}")
print("\nCluster membership for each element:")
print(tabulate(data[['Element Name', 'Cluster']], headers='keys', tablefmt='psql'))
