import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from tabulate import tabulate
data = pd.read_csv("Modern_Periodic_Table.csv")
selected_columns = ['Element Name', 'Atomic Number', 'Block', 'Period', 'Electron Configuration', 
                    'CAS Number', 'Half-Life', 'Lifetime', 'Quantum Numbers', 
                    'Graph Period', 'Graph Group',
                    'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
                    'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat',
                    'Molar Volume', 'Brinell Hardness', 'Molar Magnetic Susceptibility']
data_selected = data[selected_columns].copy()  
data_selected.fillna(0, inplace=True)
for col in data_selected.select_dtypes(include=['object']):
    data_selected[col] = pd.factorize(data_selected[col])[0]
max_finite_half_life = np.nanmax(data_selected['Half-Life'][np.isfinite(data_selected['Half-Life'])])
max_finite_lifetime = np.nanmax(data_selected['Lifetime'][np.isfinite(data_selected['Lifetime'])])
data_selected['Half-Life'].replace(np.inf, max_finite_half_life, inplace=True)
data_selected['Lifetime'].replace(np.inf, max_finite_lifetime, inplace=True)
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data_selected)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data_scaled)
plt.figure(figsize=(12, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data['Atomic Number'], cmap='viridis', alpha=0.7)
for i, txt in enumerate(data['Element Name']):
    plt.annotate(txt, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
plt.colorbar(label='Atomic Number')
plt.title('-Distributed Stochastic Neighbor Embedding (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
tsne_data = pd.concat([data, pd.DataFrame(tsne_result, columns=['t-SNE Component 1', 't-SNE Component 2'])], axis=1)
summary_stats = tsne_data[['t-SNE Component 1', 't-SNE Component 2']].describe()
print("Summary of Statistics:")
print(tabulate(summary_stats, headers='keys', tablefmt='psql'))
