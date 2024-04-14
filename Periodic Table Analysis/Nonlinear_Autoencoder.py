import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class NonlinearAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.001, batch_size=32, num_epochs=100):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = Autoencoder(input_dim, encoding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.train_losses = []
    def preprocess_data(self, data):
        data = data.dropna()
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data
    def train(self, train_loader):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                inputs, _ = data
                inputs = inputs.float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            epoch_loss = running_loss / len(train_loader)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss = {epoch_loss:.4f}")
    def fit(self, data):
        processed_data = self.preprocess_data(data)
        train_data, _ = train_test_split(processed_data, test_size=0.2, random_state=42)
        train_loader = DataLoader(TensorDataset(torch.tensor(train_data), torch.zeros(len(train_data))), batch_size=self.batch_size, shuffle=True)
        self.train(train_loader)
    def encode(self, data):
        processed_data = self.preprocess_data(data)
        with torch.no_grad():
            encoded_data = self.model.encoder(torch.tensor(processed_data).float())
        return encoded_data.numpy()
    def decode(self, encoded_data):
        with torch.no_grad():
            decoded_data = self.model.decoder(torch.tensor(encoded_data).float())
        return decoded_data.numpy()
    def plot_training_loss(self):
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
if __name__ == "__main__":
    data = pd.read_csv('Periodic_Table.csv')
    columns_of_interest = ['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point', 'Molar Heat', 'Atomic Number', 'Period', 'Group', 'X Position', 'Y Position', 'WX Position', 'WY Position', 'Electron Affinity']
    data = data[columns_of_interest]
    print(f"Dataset size: {data.shape[0]} samples, {data.shape[1]} features")
    input_dim = len(columns_of_interest)
    encoding_dim = 5
    print(f"Encoding dimension: {encoding_dim}")
    autoencoder = NonlinearAutoencoder(input_dim, encoding_dim)
    autoencoder.fit(data)
    autoencoder.plot_training_loss()
    encoded_data = autoencoder.encode(data)
    decoded_data = autoencoder.decode(encoded_data)
    print(f"Final loss = {autoencoder.train_losses[-1]:.4f}")
