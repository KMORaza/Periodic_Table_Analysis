# Temporal Convolutional Network (TCN)
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
data = pd.read_csv('Periodic_Table.csv')
data = data.dropna()
X = data.drop(columns=['Melting Point']) 
y = data['Melting Point']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1) 
])
model.compile(optimizer=Adam(), loss=MeanSquaredError())
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_test, y_test)
print("Test Loss =", loss)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Melting Point')
plt.ylabel('Predicted Melting Point')
plt.title('Actual vs Predicted Melting Point')
plt.show()
