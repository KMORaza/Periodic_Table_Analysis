# Deep Variational Information Bottleneck (DVIB)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
data = pd.read_csv("Periodic_Table.csv")
data.dropna(inplace=True)
features = data.drop(columns=['Atomic Mass'])  
target = data['Atomic Mass']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
X = torch.tensor(scaled_features, dtype=torch.float32)
y = torch.tensor(target.values, dtype=torch.float32)
class DVIB(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DVIB, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.latent_dim = latent_dim
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        enc = self.encoder(x)
        mu, log_var = enc[:, :self.latent_dim], enc[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)
        dec = self.decoder(z)
        return dec, mu, log_var
input_dim = X.shape[1]
latent_dim = 2  
model = DVIB(input_dim, latent_dim)
def dvib_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
batch_size = 64
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
total_losses = []
recon_losses = []
kl_losses = []
for epoch in range(epochs):
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs, _ = batch
        recon_batch, mu, log_var = model(inputs)
        loss = dvib_loss(recon_batch, inputs, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_loss_total += nn.functional.mse_loss(recon_batch, inputs, reduction='sum').item()
        kl_loss_total += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
    recon_loss_avg = recon_loss_total / len(dataset)
    kl_loss_avg = kl_loss_total / len(dataset)
    total_losses.append(total_loss / len(dataset))
    recon_losses.append(recon_loss_avg)
    kl_losses.append(kl_loss_avg)
    print(f"Epoch {epoch+1}/{epochs}: \nTotal Loss = {total_losses[-1]} \nRecon Loss = {recon_losses[-1]} \nKL Loss = {kl_losses[-1]}")
plt.figure(figsize=(10, 5))
plt.plot(total_losses, label='Total Loss')
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.show()
