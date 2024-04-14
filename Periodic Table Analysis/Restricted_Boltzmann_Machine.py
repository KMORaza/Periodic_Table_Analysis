import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
data = pd.read_csv('/Dataset/Modern_Periodic_Table.csv')
X = data[['Atomic Mass', 'Atomic Number', 'Period', 'X Position', 'Y Position']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rbm = BernoulliRBM(n_components=64,  
                   n_iter=200,        
                   learning_rate=0.01,
                   batch_size=10,     
                   random_state=42,  
                   verbose=True)     
rbm_pipeline = Pipeline([('scaler', scaler), ('rbm', rbm)])
rbm_pipeline.fit(X_scaled)
X_transformed = rbm_pipeline.transform(X_scaled)
print("\nTransformed Data:")
print(X_transformed)
print("\n")
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
rbm2 = BernoulliRBM(n_components=5, learning_rate=0.1, batch_size=10, n_iter=100, verbose=1, random_state=42)
rbm_pipeline2 = Pipeline(steps=[('rbm', rbm)])
rbm_pipeline2.fit(X_train)
train_score = rbm_pipeline2.score_samples(X_train)
test_score = rbm_pipeline2.score_samples(X_test)
print("\nRBM training score =", np.mean(train_score))
print("RBM testing score =", np.mean(test_score))
class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, batch_size=10, epochs=100):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def gibbs_sampling(self, visible_prob):
        hidden_prob = self.sigmoid(np.dot(visible_prob, self.weights) + self.hidden_bias)
        hidden_state = np.random.binomial(1, hidden_prob)
        visible_prob = self.sigmoid(np.dot(hidden_state, self.weights.T) + self.visible_bias)
        return hidden_state, visible_prob
    def train(self, data):
        num_samples = data.shape[0]
        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for i in range(0, num_samples, self.batch_size):
                batch_data = data[i:i+self.batch_size]
                positive_hidden_prob = self.sigmoid(np.dot(batch_data, self.weights) + self.hidden_bias)
                positive_hidden_state = np.random.binomial(1, positive_hidden_prob)
                negative_hidden_state, reconstructed_visible_prob = self.gibbs_sampling(batch_data)
                negative_visible_prob = reconstructed_visible_prob
                self.weights += self.learning_rate * (np.dot(batch_data.T, positive_hidden_prob) - np.dot(negative_visible_prob.T, negative_hidden_state)) / self.batch_size
                self.visible_bias += self.learning_rate * np.mean(batch_data - negative_visible_prob, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(positive_hidden_prob - negative_hidden_state, axis=0)
    def generate_samples(self, num_samples):
        visible_samples = np.random.rand(num_samples, self.num_visible)
        _, generated_samples = self.gibbs_sampling(visible_samples)
        return generated_samples
class PeriodicTableData:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
    def preprocess_data(self):
        self.data = self.data.select_dtypes(include=[np.number])
        self.data = self.data.dropna(axis=1)
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data = self.data.dropna(axis=0)
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        return self.data.values
if __name__ == "__main__":
    dataset = PeriodicTableData("Modern_Periodic_Table.csv")
    preprocessed_data = dataset.preprocess_data()
    print("\nPreprocessed Data Shape:", preprocessed_data.shape)
    rbm3 = RBM(num_visible=preprocessed_data.shape[1], num_hidden=100)
    rbm3.train(preprocessed_data)
    generated_samples = rbm3.generate_samples(10)
    print("\nGenerated Samples Shape:", generated_samples.shape)
    print("\nGenerated Samples:")
    print(generated_samples)
