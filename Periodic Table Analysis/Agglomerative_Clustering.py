import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = [
    'Element Name',
    'Atomic Mass',
    'Density',
    'Melting Point',
    'Boiling Point',
    'Heat of Fusion',
    'Heat of Vaporization',
    'Specific Heat',
    'Thermal Conductivity',
    'Thermal Expansion',
    'Molar Volume',
    'Atomic Radius',
    'Covalent Radius',
]
data = data[selected_columns].dropna()
element_names = data['Element Name']
data = data.drop(columns=['Element Name'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clusters = cluster.fit_predict(scaled_data)
data['Cluster'] = clusters
data['Element Name'] = element_names
cluster_details = data.groupby('Cluster').mean()
print(tabulate(cluster_details, headers='keys', tablefmt='grid'))
sns.pairplot(data=data, hue='Cluster', palette='Set1', vars=data.columns[1:-2])
plt.show()
