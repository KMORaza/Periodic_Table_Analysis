# Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
data = pd.read_csv("Modern_Periodic_Table.csv")
features = data[['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Specific Heat']]
features.fillna(features.mean(), inplace=True)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
birch = Birch(threshold=0.5, branching_factor=50)
birch.fit(scaled_features)
clusters = birch.predict(scaled_features)
data['Cluster'] = clusters
cluster_details = []
for cluster_id in sorted(set(clusters)):
    cluster_data = data[data['Cluster'] == cluster_id]
    cluster_size = len(cluster_data)
    cluster_mean = cluster_data.mean()
    cluster_std = cluster_data.std()
    cluster_details.append([f"Cluster {cluster_id}", f"Size: {cluster_size}", "Mean:", cluster_mean, "Std:", cluster_std])
print(tabulate(cluster_details, headers=['Cluster', 'Size', '', 'Mean', '', 'Std'], tablefmt='grid'))
plt.scatter(data['Atomic Mass'], data['Density'], c=data['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Atomic Mass')
plt.ylabel('Density')
plt.title('Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)')
plt.show()
