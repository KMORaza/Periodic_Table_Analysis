import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
class PeriodicTableNaiveBayes:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.data.dropna(inplace=True)  
    def preprocess(self, threshold=0.5):
        self.data['Boiling Point'] = (self.data['Boiling Point'] > threshold).astype(int)
        self.data['Density'] = (self.data['Density'] > threshold).astype(int)
        self.data['Melting Point'] = (self.data['Melting Point'] > threshold).astype(int)
        self.data['Molar Heat'] = (self.data['Molar Heat'] > threshold).astype(int)
        self.data['Electron Affinity'] = (self.data['Electron Affinity'] > threshold).astype(int)
    def train_test_split(self, test_size=0.2):
        features = ['Boiling Point', 'Density', 'Melting Point', 'Molar Heat', 'Electron Affinity']
        X = self.data[features]
        y = self.data['Period']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    def train_model(self):
        self.model = BernoulliNB()
        self.model.fit(self.X_train, self.y_train)
    def test_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
if __name__ == "__main__":
    data_file = "Periodic_Table.csv"
    model = PeriodicTableNaiveBayes(data_file)
    model.preprocess()
    model.train_test_split()
    model.train_model()
    model.test_model()
