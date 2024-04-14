import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from tabulate import tabulate
data = pd.read_csv('Modern_Periodic_Table.csv')
selected_features = ['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat']
data[selected_features] = data[selected_features].fillna(data[selected_features].mean())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_features])
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)
leaders = [data_scaled[0]]
non_leaders = np.delete(data_scaled, 0, axis=0)
threshold_distance = 2.0
clusters = {}
for idx, point in enumerate(non_leaders):
    min_distance = float('inf')
    closest_leader = None
    for leader_idx, leader in enumerate(leaders):
        dist = distance(point, leader)
        if dist < min_distance:
            min_distance = dist
            closest_leader = leader
            closest_leader_idx = leader_idx
    if min_distance <= threshold_distance:
        if closest_leader_idx in clusters:
            clusters[closest_leader_idx].append(idx)
        else:
            clusters[closest_leader_idx] = [idx]
    else:
        leaders.append(point)
table_data = []
for cluster_idx, elements in clusters.items():
    elements_list = data.iloc[elements]['Element Name'].tolist()
    table_data.append([cluster_idx + 1, len(elements), ", ".join(elements_list)])
print(tabulate(table_data, headers=["Cluster", "Number of Elements", "Elements"], tablefmt="grid", numalign="center"))
def plot_clusters(feature1, feature2):
    plt.figure(figsize=(10, 6))
    plt.title(f"Leader Clustering: {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    for cluster_idx, elements in clusters.items():
        plt.scatter(data.iloc[elements][feature1], data.iloc[elements][feature2], label=f'Cluster {cluster_idx + 1}')
    leaders_x = [leader[0] for leader in leaders]
    leaders_y = [leader[1] for leader in leaders]
    plt.scatter(leaders_x, leaders_y, color='black', marker='x', label='Leaders')
    plt.legend()
    plt.grid(True)
    plt.show()
feature1 = 'Atomic Mass'
feature2 = 'Density'
plot_clusters(feature1, feature2)
