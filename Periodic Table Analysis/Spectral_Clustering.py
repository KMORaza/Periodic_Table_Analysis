import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
columns_for_clustering = ['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Specific Heat',
                          'Heat of Fusion', 'Heat of Vaporization', 'Electronegativity', 'Atomic Radius',
                          'Covalent Radius', 'van der Waals Radius']
data.dropna(subset=columns_for_clustering, inplace=True)
X = data[columns_for_clustering]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
silhouette_scores = []
for n_clusters in range(2, 11):
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=42)
    cluster_labels = spectral_clustering.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()
optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print("Optimal number of clusters =", optimal_n_clusters)
spectral_clustering = SpectralClustering(n_clusters=optimal_n_clusters, affinity='rbf', random_state=42)
cluster_labels = spectral_clustering.fit_predict(X_scaled)
data['Cluster'] = cluster_labels
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=50)
plt.title('Spectral Clustering Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()
centroids = []
for cluster_id in range(optimal_n_clusters):
    cluster_data = X_scaled[cluster_labels == cluster_id]
    centroid = np.mean(cluster_data, axis=0)
    centroids.append(centroid)
for i, centroid in enumerate(centroids):
    print(f"Centroid of Cluster {i}:")
    print(tabulate([centroid], headers=columns_for_clustering, tablefmt='grid', floatfmt=".2f"))
    print()
print(tabulate(data[['Element Name', 'Cluster']], headers='keys', tablefmt='grid'))
