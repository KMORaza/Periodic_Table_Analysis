import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
features = ['Atomic Mass', 'Density', 'Melting Point', 'Boiling Point']
data.dropna(subset=features, inplace=True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(data_scaled)
clusters = gmm.predict(data_scaled)
data['Cluster'] = clusters
sns.pairplot(data=data, vars=features, hue='Cluster', palette='viridis')
plt.show()
cluster_means = pd.DataFrame(gmm.means_, columns=features)
print("Cluster Means:")
print(cluster_means)
print(tabulate(cluster_means, headers='keys', tablefmt='psql'))
covariances_reshaped = gmm.covariances_.reshape(n_components, len(features), len(features))
covariance_data = [cov_matrix.flatten() for cov_matrix in covariances_reshaped]
cluster_covariances = pd.DataFrame(covariance_data, columns=[f'{feat1}-{feat2}_cov' for feat1 in features for feat2 in features])
print("\nCluster Covariances:")
print(tabulate(cluster_covariances, headers='keys', tablefmt='psql'))
cluster_weights = gmm.weights_
print("\nCluster Weights:", cluster_weights)
