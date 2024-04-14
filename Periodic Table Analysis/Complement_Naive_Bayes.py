import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
class PeriodicTableCNB:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.model = None
    def load_data(self):
        self.data = pd.read_csv(self.filename)
    def preprocess_data(self):
        self.data.dropna(inplace=True)
        self.data = self.data.select_dtypes(include=[np.number])
        self.data = self.data.clip(lower=0)
    def train_test_split(self, test_size=0.2):
        X = self.data.drop(columns=['Period'])
        y = self.data['Period']  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    def train_model(self):
        self.model = ComplementNB()
        self.model.fit(self.X_train, self.y_train)
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
if __name__ == "__main__":
    pt_cnb = PeriodicTableCNB("Periodic_Table.csv")
    pt_cnb.load_data()
    pt_cnb.preprocess_data()
    pt_cnb.train_test_split()
    pt_cnb.train_model()
    pt_cnb.evaluate_model()
