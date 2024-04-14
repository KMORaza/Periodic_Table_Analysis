import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class CategoricalNaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
    def fit(self, X, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for cls, count in zip(unique_classes, class_counts):
            self.class_probabilities[cls] = count / total_samples
        for feature in X.columns:
            self.feature_probabilities[feature] = {}
            for cls in unique_classes:
                feature_value_counts = X[y == cls][feature].value_counts()
                total_samples_in_class = len(X[y == cls])
                self.feature_probabilities[feature][cls] = (feature_value_counts + 1) / (total_samples_in_class + len(feature_value_counts))
    def predict(self, X):
        predictions = []
        for _, sample in X.iterrows():
            max_prob = -1
            predicted_class = None
            for cls, class_prob in self.class_probabilities.items():
                likelihood = 1
                for feature, value in sample.items():
                    if value in self.feature_probabilities[feature][cls]:
                        likelihood *= self.feature_probabilities[feature][cls][value]
                    else:
                        likelihood *= 1 / (len(self.feature_probabilities[feature][cls]) + 1) # Laplace smoothing
                posterior = class_prob * likelihood
                if posterior > max_prob:
                    max_prob = posterior
                    predicted_class = cls
            predictions.append(predicted_class)
        return predictions
data = pd.read_csv('Periodic_Table.csv')
data.dropna(inplace=True)
le = LabelEncoder()
data['Group'] = le.fit_transform(data['Group'])
X = data.drop(columns=['Group'])
y = data['Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = CategoricalNaiveBayes()
nb_classifier.fit(X_train, y_train)
predictions = nb_classifier.predict(X_test)
accuracy = (predictions == y_test).mean()
print("Accuracy:", accuracy)
