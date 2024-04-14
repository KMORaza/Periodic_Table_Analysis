# Deep Autoencoding Gaussian Mixture Model (DAGMM)
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.mixture import GaussianMixture
class Encoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super(Encoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(z_dim, activation=None)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        z = self.dense3(x)
        return z
class Decoder(tf.keras.Model):
    def __init__(self, z_dim, hidden_dim2, hidden_dim1, output_dim):
        super(Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim1, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_dim, activation=None)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output
class DAGMM:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim, n_gmm_components):
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim2, hidden_dim1, input_dim)
        self.n_gmm_components = n_gmm_components
        self.gmm = GaussianMixture(n_components=n_gmm_components)
    def compute_energy(self, x, x_hat, z, gamma):
        recon_loss = tf.reduce_sum(tf.square(x - x_hat), axis=1)
        sample_energy = tf.reduce_sum(tf.square(z), axis=1)
        energy = tf.reduce_mean(recon_loss + gamma * sample_energy)
        return energy
    def fit(self, X_train, learning_rate=1e-3, epochs=100, batch_size=64, gamma=0.1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            idx = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[idx]
            total_batch = len(X_train) // batch_size
            for i in range(total_batch):
                batch_x = X_train_shuffled[i * batch_size:(i + 1) * batch_size]
                with tf.GradientTape() as tape:
                    z = self.encoder(batch_x)
                    x_hat = self.decoder(z)
                    loss = self.compute_energy(batch_x, x_hat, z, gamma)
                gradients = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
            print("Epoch:", epoch + 1, "Loss = ", loss.numpy())
        encoded_data = self.encoder.predict(X_train)
        self.gmm.fit(encoded_data)
    def predict(self, X_test):
        encoded_data = self.encoder.predict(X_test)
        reconstructions = self.decoder.predict(encoded_data)
        energy = np.sum(np.square(X_test - reconstructions), axis=1)
        likelihood = -self.gmm.score_samples(encoded_data)
        anomaly_score = energy + likelihood
        return anomaly_score
    def print_details(self):
        print("Number of Gaussian components:", self.n_gmm_components)
        print("Encoder Architecture:")
        print(self.encoder.summary())
        print("Decoder Architecture:")
        print(self.decoder.summary())
        print("")
        print("Means:\n", self.gmm.means_)
        print("")
        print("Covariances:\n", self.gmm.covariances_)
        print("")
        print("Weights = ", self.gmm.weights_)
data = pd.read_csv("Periodic_Table.csv")
data.dropna(inplace=True)
X = data.drop(columns=["Atomic Number", "Period", "Group", "Electron Affinity"])
input_dim = X.shape[1]
hidden_dim1 = 64
hidden_dim2 = 32
z_dim = 10
n_gmm_components = 3
dagmm = DAGMM(input_dim, hidden_dim1, hidden_dim2, z_dim, n_gmm_components)
dagmm.fit(X.values)
dagmm.print_details()
