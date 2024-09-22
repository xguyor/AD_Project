import torch
import torch.nn as nn

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # First hidden layer
            nn.ReLU(True),
            nn.Linear(128, 256),         # Second hidden layer
            nn.ReLU(True),
            nn.Linear(256, 512),        # Third hidden layer
            nn.ReLU(True),
            nn.Linear(512, output_dim), # Output layer
            nn.Sigmoid()                 # To ensure the output is between 0 and 1
        )

    def forward(self, z):
        return self.decoder(z)
