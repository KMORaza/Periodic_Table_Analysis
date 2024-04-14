# Deep Belief Network
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Periodic_Table.csv')
data.dropna(inplace=True)
X = data.drop(columns=['Atomic Mass']) 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
class DBN:
    def __init__(self, layer_sizes):
        self.model = Sequential()
        for i in range(len(layer_sizes) - 1):
            self.model.add(Dense(layer_sizes[i+1], activation='relu', input_shape=(layer_sizes[i],)))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    def predict(self, X_test):
        return self.model.predict(X_test)
layer_sizes = [X_train.shape[1], 100, 50]
dbn = DBN(layer_sizes)
dbn.train(X_train, X_train[:, 0], epochs=50)
y_pred = dbn.predict(X_test)
mse = mean_squared_error(X_test[:, 0], y_pred)
print("Mean Squared Error =", mse)
