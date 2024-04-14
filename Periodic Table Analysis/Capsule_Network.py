import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
class CapsuleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CapsuleNetwork, self).__init__()
        self.dense = nn.Linear(input_dim, 64)  
        self.capsule = nn.Linear(64, output_dim) 
    def forward(self, x):
        x = torch.relu(self.dense(x))
        x = torch.sigmoid(self.capsule(x))
        return x
class PeriodicTableDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).dropna()  
        self.scaler = StandardScaler()
        self.features = self.data.drop(columns=['Atomic Number', 'Period', 'Group']).values 
        self.features = self.scaler.fit_transform(self.features)
        self.labels = self.data[['Period', 'Group']].values 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
input_dim = 10  
output_dim = 2 
model = CapsuleNetwork(input_dim, output_dim)
dataset = PeriodicTableDataset("Periodic_Table.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_losses = []
train_accuracies = []
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for i, batch in enumerate(dataloader):
        features, labels = batch['features'], batch['labels']
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted_labels = torch.round(outputs)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.tight_layout()
plt.show()
