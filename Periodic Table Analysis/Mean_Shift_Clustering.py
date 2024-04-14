import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_features = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point']
X = data[selected_features]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X_scaled)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
data['Cluster'] = labels
sns.scatterplot(data=data, x=selected_features[0], y=selected_features[1], hue='Cluster', palette='viridis')
plt.title('Clusters in 2D space')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.legend(title='Cluster')
plt.show()
num_clusters = len(cluster_centers)
print("Number of clusters:", num_clusters)
print("\nCluster centers:")
cluster_center_table = [["Cluster {}".format(i), *center] for i, center in enumerate(cluster_centers)]
print(tabulate(cluster_center_table, headers=["Cluster", *selected_features], tablefmt="pretty"))
print("\nDetails about each cluster:")
for i in range(num_clusters):
    cluster_data = data[data['Cluster'] == i][selected_features]
    num_elements = len(cluster_data)
    print("\nCluster", i, "contains", num_elements, "elements")
    print("Mean values of features in Cluster", i, ":")
    print(tabulate([["Mean"] + cluster_data.mean().tolist()], headers=["", *selected_features], tablefmt="pretty"))
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='Cluster', y=feature)
    plt.title(f'Distribution of {feature} in each cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()
