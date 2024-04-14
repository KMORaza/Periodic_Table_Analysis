import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
class PeriodicTable:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = ['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point',
                         'Molar Heat', 'Atomic Number', 'Period', 'Group',
                         'X Position', 'Y Position', 'WX Position', 'WY Position',
                         'Electron Affinity']
        self.data.dropna(subset=self.features, inplace=True)
        self.neighbor_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.neighbor_model.fit(self.data[self.features])
    def get_similar_elements(self, element_props):
        distances, indices = self.neighbor_model.kneighbors([element_props])
        similar_elements = self.data.iloc[indices[0]]
        return similar_elements
    def predict_property(self, element_props, property_name):
        similar_elements = self.get_similar_elements(element_props)
        predicted_value = similar_elements[property_name].mean()
        return predicted_value
periodic_table = PeriodicTable('Periodic_Table.csv')
element_props = [55.85, None, 7.874, None, None, 26, None, None, None, None, None, None, None]
element_props_complete = [55.85, 3000, 7.874, 1808, 25.10, 26, 4, 8, 3, 4, 6, 10, 14]
predicted_boiling_point = periodic_table.predict_property(element_props_complete, 'Boiling Point')
print("Predicted Boiling Point for Iron =", predicted_boiling_point, "°C")
