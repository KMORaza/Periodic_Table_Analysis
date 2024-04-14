import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
class HybridNaiveBayes:
    def __init__(self):
        self.gaussian_nb = GaussianNB()
        self.multinomial_nb = MultinomialNB()
    def fit(self, X_train_gaussian, X_train_multinomial, y_train):
        self.gaussian_nb.fit(X_train_gaussian, y_train)
        self.multinomial_nb.fit(X_train_multinomial, y_train)
    def predict(self, X_test_gaussian, X_test_multinomial):
        pred_gaussian = self.gaussian_nb.predict(X_test_gaussian)
        pred_multinomial = self.multinomial_nb.predict(X_test_multinomial)
        predictions = pd.Series(pred_gaussian).combine(pd.Series(pred_multinomial), lambda x1, x2: x1 if x1 == x2 else (x1 + x2) / 2)
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        return discretizer.fit_transform(predictions.values.reshape(-1, 1))
periodic_table = pd.read_csv('Periodic_Table.csv')
periodic_table.dropna(inplace=True)
X = periodic_table.drop(columns=['Atomic Number', 'Group', 'Period'])
y = periodic_table['Group']
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_discrete = discretizer.fit_transform(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_discrete, test_size=0.2, random_state=42)
X_train_gaussian = X_train[['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point', 'Molar Heat', 'Electron Affinity']]
X_test_gaussian = X_test[['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point', 'Molar Heat', 'Electron Affinity']]
X_train_multinomial = X_train[['X Position', 'Y Position', 'WX Position', 'WY Position']]
X_test_multinomial = X_test[['X Position', 'Y Position', 'WX Position', 'WY Position']]
hybrid_nb = HybridNaiveBayes()
hybrid_nb.fit(X_train_gaussian, X_train_multinomial, y_train)
predictions = hybrid_nb.predict(X_test_gaussian, X_test_multinomial)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
