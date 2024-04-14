import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("/Dataset/Modern_Periodic_Table.csv")
X = data[['Atomic Mass', 'Atomic Number', 'Period', 'X Position', 'Y Position']]
class DivisiveClustering:
    def __init__(self):
        pass
    def fit(self, X):
        self.labels_ = np.zeros(len(X))
        self._divide(X, 0)
    def _divide(self, X, label):
        if len(X) == 1:
            self.labels_[X.index] = label
            return
        split_index = len(X) // 2
        left_data = X.iloc[:split_index]
        right_data = X.iloc[split_index:]
        self._divide(left_data, label)
        self._divide(right_data, label + 1)
divisive_cluster = DivisiveClustering()
divisive_cluster.fit(X)
print("Cluster assignments =", divisive_cluster.labels_)
num_clusters = len(set(divisive_cluster.labels_))
print("\nNumber of clusters =", num_clusters)
cluster_info = []
for i in range(num_clusters):
    cluster_indices = [index for index, label in enumerate(divisive_cluster.labels_) if label == i]
    cluster_size = len(cluster_indices)
    cluster_info.append([f"Cluster {i}", f"{cluster_size}", f"{cluster_indices}"])
print("\n📝 Cluster information:")
print(tabulate(cluster_info, headers=["Cluster", "Size", "Indices"], tablefmt="grid"))
