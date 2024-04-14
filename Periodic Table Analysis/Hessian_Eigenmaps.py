import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Modern_Periodic_Table.csv")
data.dropna(axis=1, inplace=True)
numeric_columns = data.select_dtypes(include=np.number).columns
data[numeric_columns] = data[numeric_columns].clip(lower=-1e10, upper=1e10)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_columns])
distances = pairwise_distances(scaled_data, metric="euclidean")
k = 10
squared_distances = distances ** 2
neighbors_indices = np.argpartition(squared_distances, k + 1, axis=1)[:, :k + 1]
neighbor_distances = squared_distances[np.arange(squared_distances.shape[0])[:, None], neighbors_indices]
sigma2 = np.mean(neighbor_distances[:, -1])
affinity_matrix = np.exp(-squared_distances / sigma2)
degrees = np.sum(affinity_matrix, axis=1)
D = np.diag(degrees)
L = D - affinity_matrix
H = np.dot(np.dot(np.sqrt(np.diag(degrees ** -1)), L), np.sqrt(np.diag(degrees ** -1)))
n_components = 2
_, eigenvectors = np.linalg.eigh(H)
embedding = eigenvectors[:, 1:n_components + 1]
import matplotlib.pyplot as plt
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("Hessian Eigenmaps")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
