import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
class MultinomialNaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
    def fit(self, X, y):
        class_counts = y.value_counts()
        total_samples = len(y)
        for class_label, count in class_counts.items():
            self.class_probs[class_label] = count / total_samples
        for feature in X.columns:
            self.feature_probs[feature] = {}
            for class_label in self.class_probs.keys():
                feature_values = X.loc[y == class_label, feature]
                total_feature_values = feature_values.count()
                feature_prob = {}
                for value in feature_values.unique():
                    feature_prob[value] = (feature_values == value).sum() / total_feature_values
                self.feature_probs[feature][class_label] = feature_prob
    def predict(self, X):
        predictions = []
        for _, sample in X.iterrows():
            max_prob = -1
            predicted_class = None
            for class_label, class_prob in self.class_probs.items():
                prob = 1
                for feature, value in sample.items():
                    if value in self.feature_probs[feature][class_label]:
                        prob *= self.feature_probs[feature][class_label][value]
                    else:
                        prob *= 0  
                prob *= class_prob
                if prob > max_prob:
                    max_prob = prob
                    predicted_class = class_label
            predictions.append(predicted_class)
        return predictions
data = pd.read_csv("Periodic_Table.csv")
data.dropna(inplace=True)
target_column = 'Period'
X = data.drop(columns=[target_column])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = MultinomialNaiveBayes()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
