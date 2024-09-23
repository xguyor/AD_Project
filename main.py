import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from AutoDecoder import AutoDecoder
from VariationalAutoDecoder import VariationalAutoDecoder
from utils import create_dataloaders
from evaluate import evaluate_model
from sklearn.manifold import TSNE
import torch.nn.functional as F

# Function: train_ad
def train_ad(model, train_dl, latents, optimizer, num_epochs, device):
    """Train the auto-decoder model."""
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (idx, x) in enumerate(train_dl):
            idx = idx.to(device)
            x = x.to(device).view(x.size(0), -1)  # Flatten the images

            # Forward pass
            x_rec = model(latents[idx])
            loss = torch.norm(x - x_rec) / torch.prod(torch.tensor(x.shape, device=device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"AD Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dl)}")

# Function: train_vad
def train_vad(model, train_dl, optimizer, num_epochs, device):
    """Train the variational auto-decoder model."""
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (idx, x) in enumerate(train_dl):
            idx = idx.to(device)
            x = x.to(device).view(x.size(0), -1)  # Flatten the images

            # Forward pass
            x_rec, mean, log_var = model(x)
            reconstruction_loss = F.mse_loss(x_rec, x, reduction='mean')  # Use mean to reduce loss scale
            kl_weight = 0.001  # Adjust this value to balance the KL divergence with reconstruction loss
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            kl_divergence /= x.size(0)  # Average over batch size
            loss = reconstruction_loss + kl_weight * kl_divergence  # Total loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"VAD Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dl)}")

# Function: sample_and_decode
def sample_and_decode(model, latents, device, save_file, model_type="AD"):
    """Sample random and test set latent vectors and decode them."""
    model.eval()

    # Sample 5 random latent vectors from U(0, I)
    random_latents = torch.rand(5, latents.size(1), device=device)

    # Select 5 latent vectors from the test set
    test_latents = latents[torch.randint(0, latents.size(0), (5,), device=device)]

    # Decode both sets
    with torch.no_grad():
        if model_type == "VAD":
            random_decoded = model.decode(random_latents).view(5, 28, 28).cpu()
            test_decoded = model.decode(test_latents).view(5, 28, 28).cpu()
        else:
            random_decoded = model(random_latents).view(5, 28, 28).cpu()
            test_decoded = model(test_latents).view(5, 28, 28).cpu()

    # Plot and save the results
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(test_decoded[i], cmap='gray')
        axes[0, i].set_title("Test Latent")
        axes[0, i].axis('off')

        axes[1, i].imshow(random_decoded[i], cmap='gray')
        axes[1, i].set_title("Random Latent")
        axes[1, i].axis('off')

    plt.suptitle("Test Set Latents (Top) vs. Randomly Sampled Latents (Bottom)")
    plt.savefig(save_file)
    plt.show()

# Function: visualize_latent_space_with_labels
def visualize_latent_space_with_labels(latents, labels, device, n_samples=1000, save_file="latent_space_tsne_labeled.png"):
    """Visualize latent space using t-SNE and color by labels."""
    latents_sample = latents[:n_samples].to(device)
    labels_sample = labels[:n_samples].to(device)

    latents_sample = latents_sample.detach().cpu().numpy()
    labels_sample = labels_sample.detach().cpu().numpy()

    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latents_sample)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels_sample, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space by Class Label')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_file)
    plt.show()

# Function: evaluate_model_vad (already included, but making sure)
def evaluate_model_vad(model, dataloader, optimizer, latents, num_epochs, device):
    model.eval()
    total_loss = 0
    for i, (idx, x) in enumerate(dataloader):
        idx = idx.to(device)
        x = x.to(device).view(x.size(0), -1)  # Flatten the images

        with torch.no_grad():
            x_rec, mean, log_var = model(x)  # Forward pass in VAD
            reconstruction_loss = F.mse_loss(x_rec, x, reduction='mean')
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
            loss = reconstruction_loss + 0.001 * kl_divergence  # Total VAD loss

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Function: setup_data
def setup_data(batch_size, latent_dim, device):
    """Set up the dataloaders and latent vectors."""
    train_ds, train_dl, test_ds, test_dl = create_dataloaders(batch_size=batch_size)
    latents_train = torch.randn(len(train_ds), latent_dim, requires_grad=True, device=device)
    latents_test = torch.randn(len(test_ds), latent_dim, requires_grad=True, device=device)
    return train_ds, train_dl, test_ds, test_dl, latents_train, latents_test

# Function: run_ad_pipeline
def run_ad_pipeline(train_dl, test_dl, latents_train, latents_test, train_ds, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs):
    """Run AutoDecoder training, evaluation, and visualization."""
    model = AutoDecoder(latent_dim=latent_dim).to(device)
    optimizer_train = optim.Adam(list(model.parameters()) + [latents_train], lr=learning_rate)

    # Train the AutoDecoder
    train_ad(model, train_dl, latents_train, optimizer_train, num_epochs, device)

    # Create optimizer for test latents
    optimizer_test = optim.Adam([latents_test], lr=learning_rate)
    flattened_train_dl = [(idx, x.view(x.size(0), -1)) for idx, x in train_dl]
    flattened_test_dl = [(idx, x.view(x.size(0), -1)) for idx, x in test_dl]

    # Evaluate on training and test sets
    print("Evaluating on training set (optimizing training latents only)...")
    train_loss = evaluate_model(model, flattened_train_dl, optimizer_train, latents_train, latent_optimization_epochs, device)
    print(f"Final training set loss: {train_loss}")

    print("Evaluating on test set (optimizing test latents only)...")
    test_loss = evaluate_model(model, flattened_test_dl, optimizer_test, latents_test, latent_optimization_epochs, device)
    print(f"Final test set loss: {test_loss}")

    # Sample and decode latent vectors
    sample_and_decode(model, latents_test, device, "latent_comparison_ad.png", model_type="AD")

    # Visualize latent space with labels
    train_labels = train_ds.y
    visualize_latent_space_with_labels(latents_train, train_labels, device)

# Function: run_vad_pipeline
def run_vad_pipeline(train_dl, test_dl, latents_train, latents_test, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs):
    """Run VariationalAutoDecoder training, evaluation, and visualization."""
    model_vad = VariationalAutoDecoder(latent_dim=latent_dim).to(device)
    optimizer_train_vad = optim.Adam(model_vad.parameters(), lr=learning_rate)

    # Train the VariationalAutoDecoder
    print("Training Variational AutoDecoder (VAD)...")
    train_vad(model_vad, train_dl, optimizer_train_vad, num_epochs, device)

    # Sample and decode latent vectors
    sample_and_decode(model_vad, latents_train, device, "latent_comparison_vad.png", model_type="VAD")

    # Evaluate VAD model on training and test sets
    print("Evaluating VAD on training set...")
    train_loss_vad = evaluate_model_vad(model_vad, train_dl, optimizer_train_vad, latents_train, latent_optimization_epochs, device)
    print(f"Final VAD training set loss: {train_loss_vad}")

    print("Evaluating VAD on test set...")
    test_loss_vad = evaluate_model_vad(model_vad, test_dl, optimizer_train_vad, latents_test, latent_optimization_epochs, device)
    print(f"Final VAD test set loss: {test_loss_vad}")

# Function: main
def main():
    """Main function to execute both AD and VAD pipelines."""
    # Hyperparameters
    latent_dim = 64
    num_epochs = 400
    latent_optimization_epochs = 200  # Number of epochs for optimizing train and test latents
    learning_rate = 0.001
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Set up dataset and latent vectors
    train_ds, train_dl, test_ds, test_dl, latents_train, latents_test = setup_data(batch_size, latent_dim, device)

    # #1.3.1
    # # Run AutoDecoder (AD) pipeline
    # run_ad_pipeline(train_dl, test_dl, latents_train, latents_test, train_ds, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs)
    #
    # #1.3.2
    # #1-3
    # # Run VariationalAutoDecoder (VAD) pipeline
    # run_vad_pipeline(train_dl, test_dl, latents_train, latents_test, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs)
    #

    #4
    # Normal distribution VAD
    model_vad_normal = VariationalAutoDecoder(latent_dim=64, distribution='normal').to(device)
    optimizer_normal = optim.Adam(model_vad_normal.parameters(), lr=learning_rate)
    train_vad(model_vad_normal, train_dl, optimizer_normal, num_epochs, device)

    # Uniform distribution VAD
    model_vad_uniform = VariationalAutoDecoder(latent_dim=64, distribution='uniform').to(device)
    optimizer_uniform = optim.Adam(model_vad_uniform.parameters(), lr=learning_rate)
    train_vad(model_vad_uniform, train_dl, optimizer_uniform, num_epochs, device)

if __name__ == "__main__":
    main()
