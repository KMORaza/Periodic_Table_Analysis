# Self-Organizing Feature Maps (SOFM)
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('Periodic_Table.csv', delimiter=',', skip_header=1)
data = data[~np.isnan(data).any(axis=1)]
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
input_size = data.shape[1]
output_size = 10  
learning_rate = 0.5
epochs = 10
weights = np.random.rand(output_size, input_size)
error_list = []
for epoch in range(epochs):
    total_error = 0
    for sample in data:
        distances = np.linalg.norm(weights - sample, axis=1)
        winner = np.argmin(distances)
        for i, w in enumerate(weights):
            neighborhood = np.exp(-np.linalg.norm(np.array([winner // 5, winner % 5]) - np.array([i // 5, i % 5])))
            weights[i] += learning_rate * neighborhood * (sample - w)
            total_error += np.sum((sample - weights[i]) ** 2)
    error_list.append(total_error)
    print(f'Epoch {epoch + 1}/{epochs}, Total Error: {total_error}')
print('Trained Weights:')
print(weights)
plt.plot(error_list)
plt.title('Training Error')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.show()
plt.imshow(weights, cmap='viridis', aspect='auto')
plt.title('Final State of Weights')
plt.colorbar(label='Weight Magnitude')
plt.xlabel('Input Features')
plt.ylabel('Neurons')
plt.show()
