import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
class NaiveBayesClassifier:
    def __init__(self):
        self.model = GaussianNB()
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df
def main():
    dataset = load_dataset("Periodic_Table.csv")
    X = dataset.drop(columns=['Group'])
    y = dataset['Group']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
if __name__ == "__main__":
    main()
