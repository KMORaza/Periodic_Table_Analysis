import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
class PrincipalCurve:
    def __init__(self, n_components=2, epsilon=1e-5, max_iter=100):
        self.n_components = n_components
        self.epsilon = epsilon
        self.max_iter = max_iter
    def fit(self, X):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        self.X_pca = self.pca.transform(X)
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.X_pca)
        self.points = np.copy(self.X_pca)
        for _ in range(self.max_iter):
            distances, indices = self.nbrs.kneighbors(self.points)
            target_points = self.X_pca[indices.ravel()]
            self.points += self.epsilon * (target_points - self.points)
    def transform(self, X):
        return self.points
data = pd.read_csv("Periodic_Table.csv")
X = data.drop(['Atomic Number', 'Period', 'Group', 'X Position', 'Y Position', 'WX Position', 'WY Position'], axis=1)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
model = PrincipalCurve()
model.fit(X_imputed)
X_on_curve = model.transform(X_imputed)
print(X_on_curve)
curve_data = pd.DataFrame(X_on_curve, columns=['Principal Curve Dimension 1', 'Principal Curve Dimension 2'])
print(tabulate(curve_data, headers='keys', tablefmt='pretty'))
