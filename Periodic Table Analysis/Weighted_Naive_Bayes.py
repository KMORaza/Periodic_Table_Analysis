import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
class WeightedNaiveBayes:
    def __init__(self, weights):
        self.weights = weights
        self.classifier = GaussianNB()
    def preprocess_data(self, data, target_column=None):
        data.dropna(inplace=True)
        if target_column:
            X = data.drop(columns=[target_column])  
            y = data[target_column]
        else:
            X = data
            y = None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    def train(self, data, target_column):
        X, y = self.preprocess_data(data, target_column)
        weighted_X = X * np.array(list(self.weights.values()))  
        self.classifier.fit(weighted_X, y)
    def predict(self, data):
        X, _ = self.preprocess_data(data)
        weighted_X = X * np.array(list(self.weights.values()))
        return self.classifier.predict(weighted_X)
periodic_table = pd.read_csv("Periodic_Table.csv")
weights = {
    'Atomic Mass': 1,
    'Boiling Point': 0.8,
    'Density': 0.5,
    'Melting Point': 0.8,
    'Molar Heat': 0.6,
    'Period': 0.3,
    'Group': 0.3,
    'X Position': 0.2,
    'Y Position': 0.2,
    'WX Position': 0.2,
    'WY Position': 0.2,
    'Electron Affinity': 0.7
}
wnb_classifier_group = WeightedNaiveBayes(weights)
wnb_classifier_group.train(periodic_table, target_column='Group')
wnb_classifier_period = WeightedNaiveBayes(weights)
wnb_classifier_period.train(periodic_table, target_column='Period')
example_data_group = pd.DataFrame({
    'Atomic Mass': [12.01],
    'Boiling Point': [100.0],
    'Density': [0.8],
    'Melting Point': [0.0],
    'Molar Heat': [29.0],
    'Period': [2],
    'Group': [14],
    'X Position': [2],
    'Y Position': [13],
    'WX Position': [2],
    'WY Position': [13],
    'Electron Affinity': [1.0]
})
predicted_group = wnb_classifier_group.predict(example_data_group)
print("Predicted Group:", predicted_group)
example_data_period = pd.DataFrame({
    'Atomic Mass': [12.01],
    'Boiling Point': [100.0],
    'Density': [0.8],
    'Melting Point': [0.0],
    'Molar Heat': [29.0],
    'Period': [2],
    'Group': [14],
    'X Position': [2],
    'Y Position': [13],
    'WX Position': [2],
    'WY Position': [13],
    'Electron Affinity': [1.0]
})
predicted_period = wnb_classifier_period.predict(example_data_period)
print("Predicted Period:", predicted_period)
