import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import RobustScaler
class PeriodicTableRandomProjection:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.random_projection = None
    def load_dataset(self):
        self.data = pd.read_csv(self.dataset_path)
    def preprocess_data(self):
        self.data = self.data.select_dtypes(include=np.number).dropna(axis=1)
    def scale_data(self):
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        scaler = RobustScaler()
        self.data = scaler.fit_transform(self.data)
    def fit_random_projection(self, n_components):
        self.random_projection = GaussianRandomProjection(n_components=n_components)
        self.random_projection.fit(self.data)
    def transform_data(self):
        if self.random_projection is not None:
            projected_data = self.random_projection.transform(self.data)
            return projected_data
        else:
            print("Random Projection has not been fitted yet.")
            return None
if __name__ == "__main__":
    dataset_path = "Modern_Periodic_Table.csv"
    random_projection = PeriodicTableRandomProjection(dataset_path)
    random_projection.load_dataset()
    random_projection.preprocess_data()
    random_projection.scale_data()
    random_projection.fit_random_projection(n_components=10)
    transformed_data = random_projection.transform_data()
    if transformed_data is not None:
        print("Transformed data shape = ", transformed_data.shape)
