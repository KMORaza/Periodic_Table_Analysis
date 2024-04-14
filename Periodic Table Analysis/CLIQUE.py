# Clustering in Quest (CLIQUE)
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tabulate import tabulate
from pyclustering.cluster import clique
import seaborn as sns
data = pd.read_csv("Modern_Periodic_Table.csv")
features = data[['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity', 'Bulk Modulus', 'Young\'s Modulus', 'Atomic Radius', 'Covalent Radius']]
imputer = SimpleImputer(strategy='mean')
imputed_features = imputer.fit_transform(features)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)
kmedoids = KMedoids(n_clusters=5, random_state=42)
kmedoids.fit(scaled_features)
cluster_labels = kmedoids.labels_
data['Cluster'] = cluster_labels
sns.pairplot(data=data, vars=['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point'], hue='Cluster', palette='Set1')
plt.show()
selected_data = data[['Element Name', 'Cluster']]
data_list = selected_data.values.tolist()
print(tabulate(data_list, headers=['Element Name', 'Cluster'], tablefmt='fancy_grid', numalign='center'))
parameters = data[['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point']]
imputer = SimpleImputer(strategy='mean')
imputed_features = imputer.fit_transform(parameters)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)
num_intervals = 5
density_thresholds = [1, 2, 3]
for density_threshold in density_thresholds:
    try:
        clique_instance = clique.clique(scaled_features, num_intervals, density_threshold)
        clique_instance.process()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Atomic Mass', y='Melting Point', palette='Set1')
        plt.xlabel('Atomic Mass')
        plt.ylabel('Melting Point')
        plt.title(f'Clustering In Quest (Density Threshold = {density_threshold})')
        plt.show()
    except RuntimeError as e:
        print(f"Error occurred for density threshold {density_threshold}: {e}")
