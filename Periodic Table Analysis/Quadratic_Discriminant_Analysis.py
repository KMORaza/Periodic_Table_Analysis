import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
dataset_path = "/content/Dataset/Modern_Periodic_Table.csv"
periodic_table = pd.read_csv(dataset_path)
X = periodic_table[['Atomic Mass', 'Atomic Number', 'Period', 'X Position', 'Y Position']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_means = []
for label in np.unique(cluster_labels):
    X_class = X_scaled[cluster_labels == label]
    cluster_mean = np.mean(X_class, axis=0)
    cluster_means.append(cluster_mean)
within_cluster_sums = []
for label in np.unique(cluster_labels):
    X_class = X_scaled[cluster_labels == label]
    cluster_mean = cluster_means[label]
    within_cluster_sum = np.sum((X_class - cluster_mean) ** 2)
    within_cluster_sums.append(within_cluster_sum)
for label, mean, count, within_sum in zip(np.unique(cluster_labels), cluster_means, np.bincount(cluster_labels), within_cluster_sums):
    print(f"Cluster {label}:")
    print(f"  Mean = {mean}")
    print(f"  Number of data points = {count}")
    print(f"  Within-cluster sum of squares = {within_sum}")
    print()
cov_matrices = []
for label in np.unique(cluster_labels):
    X_class = X_scaled[cluster_labels == label]
    cov_matrix = np.cov(X_class, rowvar=False)
    cov_matrices.append(cov_matrix)
fig, axes = plt.subplots(1, len(cov_matrices), figsize=(15, 5))
for i, cov_matrix in enumerate(cov_matrices):
    axes[i].imshow(cov_matrix, cmap='hot', interpolation='nearest')
    axes[i].set_title(f'Covariance Matrix Class {i+1}')
plt.tight_layout()
plt.show()
