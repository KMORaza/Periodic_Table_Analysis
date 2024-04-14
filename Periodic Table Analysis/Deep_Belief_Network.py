import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv('Modern_Periodic_Table.csv')
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
columns_to_exclude = []
for col in numeric_cols:
    if np.any(np.isinf(data[col])) or np.any(np.abs(data[col]) > 1e10):
        columns_to_exclude.append(col)
numeric_cols = numeric_cols.difference(columns_to_exclude)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_cols])  
print("Mean values:")
print(scaler.mean_)
print("Standard deviation:")
print(np.sqrt(scaler.var_))
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(X_train.shape[1], activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_test, X_test))
loss = model.evaluate(X_test, X_test)
print('Test Loss = ', loss)
history = model.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_test, X_test), verbose=0)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
