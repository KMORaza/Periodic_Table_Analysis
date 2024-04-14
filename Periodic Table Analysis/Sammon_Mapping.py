import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
data = pd.read_csv('Modern_Periodic_Table.csv')
selected_columns = ['Atomic Number', 'Period', 'Graph Period', 'Graph Group']
data_numeric = data[selected_columns].select_dtypes(include=np.number)
imputer = SimpleImputer(strategy='mean')
data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_numeric_imputed), columns=data_numeric.columns)
X = data_scaled.values
if X.size == 0:
    print("No data points available for calculation!")
else:
    mds = MDS(n_components=2, dissimilarity='euclidean')
    X_transformed = mds.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
    for i, txt in enumerate(data['Atomic Number']):
        plt.annotate(txt, (X_transformed[i, 0], X_transformed[i, 1]))
    plt.title('Sammon Mapping of Periodic Table Elements')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()
