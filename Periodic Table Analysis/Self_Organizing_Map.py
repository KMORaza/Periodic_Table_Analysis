import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from minisom import MiniSom
class SOM:
    def __init__(self, input_size, map_size):
        self.input_size = input_size
        self.map_size = map_size
        self.weights = np.random.random((map_size[0], map_size[1], input_size))
    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    def find_best_matching_unit(self, x):
        min_dist = np.inf
        bmu = None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = self.euclidean_distance(x, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu = np.array([i, j])
        return bmu
    def update_weights(self, x, bmu, learning_rate, sigma):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                weight = self.weights[i, j]
                dist_to_bmu = self.euclidean_distance(bmu, np.array([i, j]))
                influence = np.exp(- (dist_to_bmu ** 2) / (2 * sigma ** 2))
                self.weights[i, j] += learning_rate * influence * (x - weight)
    def train(self, data, epochs, learning_rate=0.1, initial_sigma=1.0):
        for epoch in range(1, epochs + 1):
            sigma = initial_sigma * np.exp(-epoch / epochs)
            for sample in data:
                bmu = self.find_best_matching_unit(sample)
                self.update_weights(sample, bmu, learning_rate, sigma)
            print(f"Epoch {epoch}/{epochs}")
    def predict(self, data):
        predicted = []
        for sample in data:
            bmu = self.find_best_matching_unit(sample)
            predicted.append(bmu)
        return np.array(predicted)
data = pd.read_csv('Modern_Periodic_Table.csv')
columns_for_som = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity', 'Thermal Expansion', 'Liquid Density', 'Molar Volume', 'Brinell Hardness', 'Mohs Hardness', 'Vickers Hardness', 'Bulk Modulus', 'Shear Modulus', "Young's Modulus", "Poisson's Ratio", 'Refractive Index', 'Speed of Sound', 'Valency', 'Electronegativity', 'Electron Affinity', 'Autoignition Point', 'Flash Point', 'Atomic Radius', 'Covalent Radius', 'van der Waals Radius']
imputer = SimpleImputer(strategy='mean')
data[columns_for_som] = imputer.fit_transform(data[columns_for_som])
try:
    input_size = len(columns_for_som)
    map_size = (10, 10)
    epochs = 100
    data_normalized = (data[columns_for_som] - data[columns_for_som].mean()) / data[columns_for_som].std()
    som = SOM(input_size, map_size)
    som.train(data_normalized.values, epochs)
    predicted = som.predict(data_normalized.values)
    plt.figure(figsize=(map_size[0], map_size[1]))
    plt.scatter(predicted[:, 0], predicted[:, 1])
    plt.title('Self-Organizing Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
except Exception as e:
    print("Error:", e)
columns_for_som = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat', 'Thermal Conductivity', 'Thermal Expansion', 'Liquid Density', 'Molar Volume', 'Brinell Hardness', 'Mohs Hardness', 'Vickers Hardness', 'Bulk Modulus', 'Shear Modulus', "Young's Modulus", "Poisson's Ratio", 'Refractive Index', 'Speed of Sound', 'Valency', 'Electronegativity', 'Electron Affinity', 'Autoignition Point', 'Flash Point', 'Atomic Radius', 'Covalent Radius', 'van der Waals Radius']
data[columns_for_som] = data[columns_for_som].fillna(data[columns_for_som].mean())
data_normalized = (data[columns_for_som] - data[columns_for_som].mean()) / data[columns_for_som].std()
size_of_map = 10
epochs = 100
input_len = len(columns_for_som)
som = MiniSom(size_of_map, size_of_map, input_len, sigma=0.5, learning_rate=0.5)
som.random_weights_init(data_normalized.values)
som.train_batch(data_normalized.values, epochs, verbose=True)
plt.figure(figsize=(size_of_map, size_of_map))
for i, x in enumerate(data_normalized.values):
    w = som.winner(x)
    plt.plot(w[0]+.5, w[1]+.5, 'o', markerfacecolor='None', markeredgecolor='C0', markersize=10, markeredgewidth=2)
plt.axis([0, size_of_map, 0, size_of_map])
plt.show()
