import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from tabulate import tabulate
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
X = data[['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization',
          'Heat of Combustion', 'Specific Heat', 'Thermal Conductivity', 'Thermal Expansion', 'Liquid Density',
          'Molar Volume', 'Brinell Hardness', 'Mohs Hardness', 'Vickers Hardness', 'Bulk Modulus', 'Shear Modulus',
          "Young's Modulus", "Poisson's Ratio", 'Refractive Index', 'Speed of Sound', 'Atomic Radius', 'Covalent Radius',
          'van der Waals Radius']]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
aff_prop = AffinityPropagation(damping=0.7)
aff_prop.fit(X_scaled)
cluster_labels = aff_prop.labels_
data['Cluster'] = cluster_labels
cluster_assignments = data[['Element Name', 'Cluster']]
print("\nCluster assignments:")
print(tabulate(cluster_assignments, headers="keys", tablefmt="grid"))
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
table = [["Silhouette Score:", silhouette_avg]]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
cluster_counts = np.bincount(cluster_labels)
centroids = []
for cluster_label in set(cluster_labels):
    centroid = X_scaled[cluster_labels == cluster_label].mean(axis=0)
    centroids.append(centroid)
cluster_details = []
for i, centroid in enumerate(centroids):
    cluster_detail = [f"Cluster {i}", cluster_counts[i], centroid]
    cluster_details.append(cluster_detail)
print("\nCluster Details:")
print(tabulate(cluster_details, headers=["Cluster", "Number of Elements", "Centroid"], tablefmt="grid"))
plt.figure(figsize=(10, 6))
for cluster_label in set(cluster_labels):
    cluster_data = X_scaled[cluster_labels == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')
plt.title('Cluster Assignments')
plt.xlabel('Atomic Mass (Scaled)')
plt.ylabel('Density (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
