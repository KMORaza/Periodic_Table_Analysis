# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
import warnings
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tabulate import tabulate
warnings.filterwarnings("ignore", message="The default value of numeric_only in DataFrame.mean is deprecated")
data = pd.read_csv("Modern_Periodic_Table.csv")
features = ['Density', 'Melting Point', 'Boiling Point', 'Atomic Mass', 'Heat of Fusion', 'Heat of Vaporization']
data.dropna(subset=features, inplace=True)
data.fillna(data.mean(), inplace=True)
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
print('Estimated number of clusters = %d' % n_clusters_)
print('Estimated number of noise points = %d' % n_noise_)
for cluster_label in set(clusters):
    cluster_data = data[clusters == cluster_label]
    print(f"\nCluster {cluster_label}:")
    print(f"Number of elements: {len(cluster_data)}")
    print(tabulate(cluster_data.describe(), headers='keys', tablefmt='grid'))
    print("\n")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='.')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
for cluster_label in set(clusters):
    cluster_data = data[clusters == cluster_label]
    print(f"Cluster {cluster_label}:")
    print(tabulate(cluster_data.describe(), headers='keys', tablefmt='grid')) 
    print("\n")
fig, axs = plt.subplots(len(features), len(features), figsize=(15, 15))
for i in range(len(features)):
    for j in range(len(features)):
        if i != j:
            axs[i, j].scatter(X_scaled[:, i], X_scaled[:, j], c=clusters, cmap='viridis', marker='.')
            axs[i, j].set_xlabel(features[i])
            axs[i, j].set_ylabel(features[j])
        else:
            axs[i, j].text(0.5, 0.5, features[i], fontsize=12, ha='center', va='center')
plt.tight_layout()
plt.show()
