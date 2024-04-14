# Averaged One-Dependence Estimator (AODE)
import pandas as pd
import numpy as np
class AODE:
    def __init__(self):
        self.probability_tables = {}
    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        data.dropna(inplace=True)
        class_counts = data[y.name].value_counts()
        total_samples = data.shape[0]
        class_probs = class_counts / total_samples
        for feature in X.columns:
            conditional_probs = data.groupby([y.name, feature]).size().unstack(fill_value=0) + 1
            conditional_probs = conditional_probs.div(conditional_probs.sum(axis=1), axis=0)  
            self.probability_tables[feature] = np.log(conditional_probs)
        self.class_probs = np.log(class_probs)
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_scores = []
            for class_val in self.class_probs.index:
                score = self.class_probs.loc[class_val]
                for feature in X.columns:
                    value = row[feature]
                    if value in self.probability_tables[feature].columns:
                        score += self.probability_tables[feature].loc[class_val, value]
                    else:
                        score += np.log(1 / (self.probability_tables[feature].shape[1] + 1))  # Laplace smoothing
                class_scores.append(score)
            predicted_class = self.class_probs.index[np.argmax(class_scores)]
            predictions.append(predicted_class)
        return predictions
data = pd.read_csv('Periodic_Table.csv')
data.dropna(inplace=True)
X = data.drop(columns=['Group'])  
y = data['Group']  
aode_classifier = AODE()
aode_classifier.fit(X, y)
predictions = aode_classifier.predict(X.head())  
print(predictions)
