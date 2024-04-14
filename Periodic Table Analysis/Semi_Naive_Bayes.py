import pandas as pd
import numpy as np
class SemiNaiveBayes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_probs = {}
        self.feature_probs = {}
    def fit(self, target_column):
        self.class_probs = self.calculate_class_probabilities(target_column)
        self.feature_probs = self.calculate_feature_probabilities(target_column)
    def calculate_class_probabilities(self, target_column):
        class_counts = self.dataset[target_column].value_counts()
        total_samples = len(self.dataset)
        class_probs = {}
        for class_label, count in class_counts.items():
            class_probs[class_label] = count / total_samples
        return class_probs
    def calculate_feature_probabilities(self, target_column):
        feature_probs = {}
        for feature in self.dataset.columns:
            if feature != target_column:
                feature_probs[feature] = {}
                for class_label in self.class_probs.keys():
                    class_subset = self.dataset[self.dataset[target_column] == class_label]
                    feature_probs[feature][class_label] = class_subset[feature].mean()
        return feature_probs
    def predict(self, sample):
        predictions = {}
        for class_label, class_prob in self.class_probs.items():
            likelihood = 1
            for feature, value in sample.items():
                if feature in self.feature_probs:
                    feature_prob = self.feature_probs[feature][class_label]
                    likelihood *= self.normal_distribution_probability(value, feature_prob)
            predictions[class_label] = likelihood * class_prob
        return max(predictions, key=predictions.get)
    def normal_distribution_probability(self, x, mean, std=0.1):
        exponent = np.exp(-((x-mean)**2 / (2 * std**2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
dataset = pd.read_csv("Periodic_Table.csv").dropna()
target_column = "Group"
semi_naive_bayes = SemiNaiveBayes(dataset)
semi_naive_bayes.fit(target_column)
sample = {
    "Atomic Mass": 12.01,
    "Boiling Point": 3530,
    "Density": 2.7,
    "Melting Point": 3823,
    "Molar Heat": 24.440,
    "Atomic Number": 6,
    "Period": 2,
    "X Position": 14,
    "Y Position": 2,
    "WX Position": 14,
    "WY Position": 2,
    "Electron Affinity": 122.26
}
prediction = semi_naive_bayes.predict(sample)
print("Predicted group:", prediction)
