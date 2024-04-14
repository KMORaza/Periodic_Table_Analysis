# Deep Belief Network
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("Periodic_Table.csv")
data.dropna(inplace=True)
features = data.drop(columns=['Atomic Number'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['Atomic Number'], test_size=0.2, random_state=42)
class RBM(tf.keras.Model):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = tf.Variable(tf.random.normal([num_visible, num_hidden], stddev=0.1))
        self.visible_bias = tf.Variable(tf.zeros([num_visible]))
        self.hidden_bias = tf.Variable(tf.zeros([num_hidden]))
    def call(self, inputs):
        hidden_prob = tf.nn.sigmoid(tf.matmul(inputs, self.W) + self.hidden_bias)
        hidden_state = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob))))
        visible_prob = tf.nn.sigmoid(tf.matmul(hidden_state, tf.transpose(self.W)) + self.visible_bias)
        return visible_prob
class DBN(tf.keras.Model):
    def __init__(self, rbm_layers):
        super(DBN, self).__init__()
        self.rbm_layers = rbm_layers
    def call(self, inputs):
        x = inputs
        for rbm in self.rbm_layers:
            x = rbm(x)
        return x
num_visible = X_train.shape[1]
num_hidden_units = [256, 128]  
rbm_layers = []
for num_hidden in num_hidden_units:
    rbm_layers.append(RBM(num_visible, num_hidden))
dbn = DBN(rbm_layers)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)  
for i, rbm in enumerate(rbm_layers):
    print(f"\nTraining layer {i+1} of Restricted Boltzmann Machine")
    for epoch in range(50):
        with tf.GradientTape() as tape:
            reconstructed = rbm(X_train)
            loss = tf.reduce_mean(tf.square(X_train - reconstructed))
        gradients = tape.gradient(loss, rbm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rbm.trainable_variables))
        print(f"Epoch-{epoch+1}: Loss = {loss:.4f}")
reconstructed = dbn(X_test)
loss = tf.reduce_mean(tf.square(X_test - reconstructed))
print("\nTest Loss =", loss.numpy())
