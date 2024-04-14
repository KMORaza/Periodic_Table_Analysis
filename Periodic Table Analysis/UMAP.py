# Uniform Manifold Approximation and Projection (UMAP)
import pandas as pd
import matplotlib.pyplot as plt
import umap
data = pd.read_csv("Modern_Periodic_Table.csv")
data_numeric = data.select_dtypes(include=['float64', 'int64'])
data_numeric.fillna(data_numeric.median(), inplace=True)
data_numeric = data_numeric.clip(lower=-1e9, upper=1e9)
if data_numeric.shape[0] > 0:
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data_numeric)
    print("UMAP parameters:")
    print("Number of components:", reducer.n_components)
    print("Number of neighbors:", reducer.n_neighbors)
    print("Minimum distance:", reducer.min_dist)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.title('UMAP projection of Periodic Table')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()
else:
    print("The dataset is empty after preprocessing!")
