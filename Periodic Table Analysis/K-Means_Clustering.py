import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
numerical_data = data[['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                       'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity',
                       'Electronegativity', 'Atomic Radius', 'Covalent Radius']]
imputer = SimpleImputer(strategy='mean')
numerical_data_imputed = imputer.fit_transform(numerical_data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data_imputed)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, clusters)
    silhouette_scores.append(silhouette_avg)
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.tight_layout()
plt.show()
best_k_elbow = inertia.index(min(inertia)) + 1
best_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
best_k = max(best_k_elbow, best_k_silhouette)
print("Elbow Method - Inertia Values:")
for k, inertia_value in enumerate(inertia, 1):
    print(f"Number of clusters: {k}, Inertia: {inertia_value}")
print("\nSilhouette Score Method - Silhouette Scores:")
for k, silhouette_score_value in enumerate(silhouette_scores, 2):
    print(f"Number of clusters: {k}, Silhouette Score: {silhouette_score_value}")
print("\nBest number of clusters (Elbow Method) =", best_k_elbow)
print("Best number of clusters (Silhouette Score Method) =", best_k_silhouette)
print("Best number of clusters (Overall) =", best_k)
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
data['Cluster'] = clusters
print("\nCount of elements in each cluster:")
cluster_counts = data['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
cluster_counts_table = tabulate(cluster_counts, headers='keys', tablefmt='pretty', showindex=False)
print(cluster_counts_table)
print("Centroids of clusters:")
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=numerical_data.columns)
print(tabulate(centroids_df, headers='keys', tablefmt='pretty'))
print()
cluster_stats = data.groupby('Cluster').agg({
    'Atomic Number': ['min', 'max', 'mean'],
    'Atomic Mass': ['min', 'max', 'mean'],
    'Density': ['min', 'max', 'mean'],
    'Melting Point': ['min', 'max', 'mean'],
    'Boiling Point': ['min', 'max', 'mean'],
    'Heat of Fusion': ['min', 'max', 'mean'],
    'Heat of Vaporization': ['min', 'max', 'mean'],
    'Specific Heat': ['min', 'max', 'mean'],
    'Thermal Conductivity': ['min', 'max', 'mean'],
    'Electronegativity': ['min', 'max', 'mean'],
    'Atomic Radius': ['min', 'max', 'mean'],
    'Covalent Radius': ['min', 'max', 'mean']
})
print("Statistics for each cluster:")
print(tabulate(cluster_stats, headers='keys', tablefmt='pretty'))
print()
print("Cluster Labels for each data point:")
cluster_labels_table = tabulate(data[['Element Name', 'Cluster']], headers='keys', showindex=False, tablefmt='pretty')
print(cluster_labels_table)
plt.scatter(data['Atomic Number'], data['Melting Point'], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Atomic Number')
plt.ylabel('Melting Point')
plt.title('K-means Clustering')
plt.legend()
plt.show()
