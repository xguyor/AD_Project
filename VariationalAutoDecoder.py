import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoDecoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=784, distribution='none'):
        super(VariationalAutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution  # Set the default distribution

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mean = nn.Linear(256, latent_dim)  # Outputs mean of the latent space
        self.fc_log_var = nn.Linear(256, latent_dim)  # Outputs log variance of the latent space

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, input_dim)  # Outputs reconstructed image (784 = 28x28)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        if self.distribution == 'normal':
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std  # Normal distribution reparameterization
        elif self.distribution == 'uniform':
            # Uniform distribution: Draw from a range between -sqrt(3)*std and sqrt(3)*std
            std = torch.exp(0.5 * log_var)
            eps = (torch.rand_like(std) * 2 - 1) * (std * (3 ** 0.5))
            return mean + eps
        else:
            # If no distribution is selected, return mean as latent space
            return mean

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))  # Ensure output is between 0 and 1 for pixel values

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_rec = self.decode(z)
        return x_rec, mean, log_var
