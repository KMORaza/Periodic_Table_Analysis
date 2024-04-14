import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = pd.read_csv('Periodic_Table.csv')
features = ['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point', 'Molar Heat', 
            'Atomic Number', 'Period', 'Group', 'X Position', 'Y Position', 
            'WX Position', 'WY Position', 'Electron Affinity']
data.fillna(data.mean(), inplace=True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_size)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
input_size = len(features)
encoding_dim = 5 
model = Autoencoder(input_size, encoding_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(data_tensor)
    loss = criterion(outputs, data_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), losses, color='red')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
encoded_features = model.encoder(data_tensor).detach().numpy()
encoded_mean = np.mean(encoded_features, axis=0)
encoded_std = np.std(encoded_features, axis=0)
plt.figure(figsize=(8, 6))
for i in range(encoding_dim):
    plt.subplot(2, 3, i+1)
    plt.hist(encoded_features[:, i], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Encoded Feature {i+1}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
print("Encoded Features:")
print(encoded_features)
print("Encoded Features (First 5 rows):")
print(encoded_features[:5])
print("\nMean of Encoded Features:")
print(encoded_mean)
print("\nStandard Deviation of Encoded Features:")
print(encoded_std)
