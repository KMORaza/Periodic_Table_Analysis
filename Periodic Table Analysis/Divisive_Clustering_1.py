import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DivisiveClustering:
    def __init__(self, min_cluster_size):
        self.min_cluster_size = min_cluster_size
    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0])
        self._split_clusters(X, np.arange(X.shape[0]))
    def _split_clusters(self, X, cluster_indices):
        if len(cluster_indices) < self.min_cluster_size:
            return
        max_diameter = -1
        for cluster_index in cluster_indices:
            cluster_points = X[cluster_index]
            cluster_diameter = np.max(np.linalg.norm(cluster_points - np.mean(cluster_points, axis=0)))
            if cluster_diameter > max_diameter:
                max_diameter = cluster_diameter
                max_diameter_index = cluster_index
        new_clusters_indices = []
        for index in cluster_indices:
            if index != max_diameter_index:
                dist_to_split_point = np.linalg.norm(X[index] - X[max_diameter_index])
                if dist_to_split_point < max_diameter / 2:
                    new_clusters_indices.append(index)
        new_label = np.max(self.labels_) + 1
        for index in new_clusters_indices:
            self.labels_[index] = new_label
        self._split_clusters(X, new_clusters_indices)
data = pd.read_csv("/Dataset/Modern_Periodic_Table.csv")
X = data[['Atomic Mass', 'Atomic Number', 'Period', 'X Position', 'Y Position']]
min_cluster_size = 5  
divisive_clustering = DivisiveClustering(min_cluster_size)
divisive_clustering.fit(X.values)
plt.scatter(X['X Position'], X['Y Position'], c=divisive_clustering.labels_)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Divisive Clustering')
plt.show()
