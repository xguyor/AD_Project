import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from AutoDecoder import AutoDecoder
from VariationalAutoDecoder import VariationalAutoDecoder
from utils import create_dataloaders
from evaluate import evaluate_model
from sklearn.manifold import TSNE
import torch.nn.functional as F
import numpy as np


def slerp(val, low, high):
    """Spherical interpolation between low and high"""
    low = low.squeeze(0)  # Remove batch dimension if necessary
    high = high.squeeze(0)  # Remove batch dimension if necessary
    omega = torch.acos(torch.dot(low / torch.norm(low), high / torch.norm(high)))
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / sin_omega) * low + (torch.sin(val * omega) / sin_omega) * high


def interpolate_latent_vectors_slerp(latent1, latent2, num_interpolations=5):
    """Perform SLERP (Spherical Linear Interpolation) between two latent vectors."""
    ratios = torch.linspace(0, 1, steps=num_interpolations + 2).to(latent1.device)  # Include 0 and 1
    interpolations = [slerp(ratio, latent1, latent2) for ratio in ratios]
    return interpolations


def interpolate_and_plot_slerp(model, sample1, sample2, device, save_file="interpolations_slerp.png"):
    """Interpolate latent vectors between two samples using SLERP and plot the decoded images."""
    model.eval()

    # Extract latent vectors for both samples
    latent1 = extract_latent_vectors_from_sample(model, sample1, device)
    latent2 = extract_latent_vectors_from_sample(model, sample2, device)

    # Perform SLERP interpolation (5 interpolations)
    interpolations = interpolate_latent_vectors_slerp(latent1, latent2, num_interpolations=5)

    # Decode original samples and interpolations
    with torch.no_grad():
        decoded_images = [model.decode(latent).view(28, 28).cpu().numpy() for latent in interpolations]

        # Decode the original latent vectors (sample1 and sample2)
        decoded_original1 = model.decode(latent1).view(28, 28).cpu().numpy()
        decoded_original2 = model.decode(latent2).view(28, 28).cpu().numpy()

    # Plot the results (original1, interpolations, original2)
    plot_interpolations(decoded_original1, decoded_images, decoded_original2, save_file)


def plot_interpolations(original1, interpolations, original2, save_file="interpolations.png"):
    """Plot original vectors and interpolated vectors."""
    num_interpolations = len(interpolations)
    fig, axes = plt.subplots(1, num_interpolations + 2, figsize=(15, 3))  # +2 for the original images

    # Plot the original sample 1
    axes[0].imshow(original1, cmap='gray')
    axes[0].set_title('Original 1')
    axes[0].axis('off')

    # Plot the interpolated images
    for i, img in enumerate(interpolations):
        axes[i + 1].imshow(img, cmap='gray')
        axes[i + 1].axis('off')

    # Plot the original sample 2
    axes[num_interpolations + 1].imshow(original2, cmap='gray')
    axes[num_interpolations + 1].set_title('Original 2')
    axes[num_interpolations + 1].axis('off')

    plt.suptitle(f"Interpolation from Original 1 to Original 2 {save_file}")
    plt.savefig(save_file)
    plt.show()

def extract_latent_vectors_from_sample(model, x, device):
    """Extract latent vectors for a given sample from the model."""
    x = x.to(device).view(-1, 28*28).float()  # Flatten the image to 784 dimensions (28x28)
    mean, log_var = model.encode(x)  # Forward pass through encoder
    z = model.reparameterize(mean, log_var)  # Reparameterization trick
    return z

def extract_latent_vectors(model, dataloader, device):
    """Extract latent vectors from the model for t-SNE visualization."""
    model.eval()
    latents = []

    with torch.no_grad():
        for i, (idx, x) in enumerate(dataloader):
            x = x.to(device).view(x.size(0), -1)  # Flatten the images
            mean, log_var = model.encode(x)
            z = model.reparameterize(mean, log_var)  # Latent vectors after reparameterization
            latents.append(z.cpu().numpy())  # Collect latent vectors

    return np.concatenate(latents, axis=0)  # Return all latents as a single array

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
def visualize_latent_space_distribution(latents, device, save_file, n_samples=1000):
    """Visualize latent space distribution using t-SNE and color based on vector norms."""

    # Ensure latents is a PyTorch tensor and convert it to the device
    if isinstance(latents, np.ndarray):
        latents_sample = torch.tensor(latents[:n_samples]).to(device)
    else:
        latents_sample = latents[:n_samples].to(device)

    # Convert to NumPy only after selecting from tensor
    latents_sample = latents_sample.detach().cpu().numpy()

    # Calculate norms of latent vectors to color them based on magnitude
    norms = np.linalg.norm(latents_sample, axis=1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latents_sample)

    # Plot with color representing norms
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=norms, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label="Latent Vector Norms")
    plt.title(f't-SNE of Latent Space Distribution {save_file}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_file)
    plt.show()


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
    visualize_latent_space_distribution(latents_train,  device,save_file="latent_space_distribution_ad.png")

# Function: run_vad_pipeline
def run_vad_pipeline(train_dl, test_dl, latents_train, latents_test, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs):
    """Run VariationalAutoDecoder pipeline."""
    model_vad = VariationalAutoDecoder(latent_dim=latent_dim).to(device)
    optimizer_train_vad = optim.Adam(model_vad.parameters(), lr=learning_rate)

    # Train the VariationalAutoDecoder
    print("Training Variational AutoDecoder (VAD)...")
    train_vad(model_vad, train_dl, optimizer_train_vad, num_epochs, device)

    # Get the decoded latents for VAD (encoding and decoding real data, not latents)
    with torch.no_grad():
        latents_train_mean, latents_train_log_var = [], []
        for _, x in train_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, log_var = model_vad.encode(x)
            latents_train_mean.append(mean)
            latents_train_log_var.append(log_var)
        train_latents_decoded = model_vad.decode(torch.cat(latents_train_mean))

    with torch.no_grad():
        latents_test_mean, latents_test_log_var = [], []
        for _, x in test_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, log_var = model_vad.encode(x)
            latents_test_mean.append(mean)
            latents_test_log_var.append(log_var)
        test_latents_decoded = model_vad.decode(torch.cat(latents_test_mean))

    # Flatten x in the dataloader inside evaluate_model
    print("Evaluating VAD on training set...")
    train_loss_vad = evaluate_model(
        lambda latents: model_vad.decode(latents),  # Ensure `evaluate_model` only gets `x_rec`
        [(i, x.view(x.size(0), -1).to(device)) for i, x in train_dl],  # Flatten x here
        optimizer_train_vad,
        torch.cat(latents_train_mean),
        latent_optimization_epochs,
        device
    )
    print(f"Final VAD training set loss: {train_loss_vad}")

    print("Evaluating VAD on test set...")
    test_loss_vad = evaluate_model(
        lambda latents: model_vad.decode(latents),  # Ensure `evaluate_model` only gets `x_rec`
        [(i, x.view(x.size(0), -1).to(device)) for i, x in test_dl],  # Flatten x here
        optimizer_train_vad,
        torch.cat(latents_test_mean),
        latent_optimization_epochs,
        device
    )
    print(f"Final VAD test set loss: {test_loss_vad}")

    # Sample and decode latent vectors for visualization
    sample_and_decode(model_vad, torch.cat(latents_test_mean), device, "latent_comparison_vad.png", model_type="VAD")

    # Visualize latent space with labels
    visualize_latent_space_distribution(torch.cat(latents_train_mean), device, save_file="latent_space_distribution_vad.png")


def gaussian_vad(model_vad_gaussian, optimizer_gaussian, train_dl, test_dl, latent_dim, device, num_epochs,
                 learning_rate, latents_train, latents_test, train_labels):
    ################### Gaussian VAD ########################

    # Get the decoded latents for VAD (encoding and decoding real data, not latents)
    with torch.no_grad():
        latents_train_mean = []
        for _, x in train_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, _ = model_vad_gaussian.encode(x)
            latents_train_mean.append(mean)
        latents_train_mean = torch.cat(latents_train_mean)

    with torch.no_grad():
        latents_test_mean = []
        for _, x in test_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, _ = model_vad_gaussian.encode(x)
            latents_test_mean.append(mean)
        latents_test_mean = torch.cat(latents_test_mean)

    print("Evaluating VAD with Gaussian distribution on training set...")
    train_loss_gaussian = evaluate_model(
        lambda latents: model_vad_gaussian.decode(latents),
        [(i, x.view(x.size(0), -1).to(device)) for i, x in train_dl],  # Flatten x here
        optimizer_gaussian,
        latents_train_mean,
        num_epochs,
        device
    )
    print(f"Final Gaussian VAD training set loss: {train_loss_gaussian}")

    print("Evaluating VAD with Gaussian distribution on test set...")
    test_loss_gaussian = evaluate_model(
        lambda latents: model_vad_gaussian.decode(latents),
        [(i, x.view(x.size(0), -1).to(device)) for i, x in test_dl],  # Flatten x here
        optimizer_gaussian,
        latents_test_mean,
        num_epochs,
        device
    )
    print(f"Final Gaussian VAD test set loss: {test_loss_gaussian}")

    print("Sampling and decoding latent vectors for Gaussian VAD...")
    sample_and_decode(model_vad_gaussian, latents_test_mean, device, "latent_comparison_gaussian_vad.png",
                      model_type="VAD")

    # Visualize latent space with labels
    latents = extract_latent_vectors(model_vad_gaussian, train_dl, device)
    visualize_latent_space_distribution(latents, device, save_file="latent_space_gaussian.png")


def uniform_vad(model_vad_uniform, optimizer_uniform, train_dl, test_dl, latent_dim, device, num_epochs, learning_rate,
                latents_train, latents_test, train_labels):
    ################### Uniform VAD ########################

    # Get the decoded latents for VAD (encoding and decoding real data, not latents)
    with torch.no_grad():
        latents_train_mean = []
        for _, x in train_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, _ = model_vad_uniform.encode(x)
            latents_train_mean.append(mean)
        latents_train_mean = torch.cat(latents_train_mean)

    with torch.no_grad():
        latents_test_mean = []
        for _, x in test_dl:
            x = x.to(device).view(x.size(0), -1)  # Flatten the images to [batch_size, 784]
            mean, _ = model_vad_uniform.encode(x)
            latents_test_mean.append(mean)
        latents_test_mean = torch.cat(latents_test_mean)

    print("Evaluating VAD with Uniform distribution on training set...")
    train_loss_uniform = evaluate_model(
        lambda latents: model_vad_uniform.decode(latents),
        [(i, x.view(x.size(0), -1).to(device)) for i, x in train_dl],  # Flatten x here
        optimizer_uniform,
        latents_train_mean,
        num_epochs,
        device
    )
    print(f"Final Uniform VAD training set loss: {train_loss_uniform}")

    print("Evaluating VAD with Uniform distribution on test set...")
    test_loss_uniform = evaluate_model(
        lambda latents: model_vad_uniform.decode(latents),
        [(i, x.view(x.size(0), -1).to(device)) for i, x in test_dl],  # Flatten x here
        optimizer_uniform,
        latents_test_mean,
        num_epochs,
        device
    )
    print(f"Final Uniform VAD test set loss: {test_loss_uniform}")

    print("Sampling and decoding latent vectors for Uniform VAD...")
    sample_and_decode(model_vad_uniform, latents_test_mean, device, "latent_comparison_uniform_vad.png",
                      model_type="VAD")

    # Visualize latent space with labels
    latents = extract_latent_vectors(model_vad_uniform, train_dl, device)
    visualize_latent_space_distribution(latents, device, save_file="latent_space_uniform.png")


# Function: main
def main():
    """Main function to execute both AD and VAD pipelines."""
    # Hyperparameters
    latent_dim = 64
    num_epochs = 100
    latent_optimization_epochs = 20  # Number of epochs for optimizing train and test latents
    learning_rate = 0.001
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Set up dataset and latent vectors
    train_ds, train_dl, test_ds, test_dl, latents_train, latents_test = setup_data(batch_size, latent_dim, device)
    train_labels = train_ds.y

    #1.3.1
    # Run AutoDecoder (AD) pipeline
    run_ad_pipeline(train_dl, test_dl, latents_train, latents_test, train_ds, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs)

    #1.3.2
    #1-3
    # Run VariationalAutoDecoder (VAD) pipeline
    run_vad_pipeline(train_dl, test_dl, latents_train, latents_test, device, latent_dim, num_epochs, learning_rate, latent_optimization_epochs)


    #4
    # gaussian distribution VAD
    train_ds, train_dl, test_ds, test_dl, latents_train, latents_test = setup_data(batch_size, latent_dim, device)
    model_vad_gaussian = VariationalAutoDecoder(latent_dim=64, distribution='gaussian').to(device)
    optimizer_gaussian = optim.Adam(model_vad_gaussian.parameters(), lr=learning_rate)
    print("Training VAD with Gaussian distribution...")
    train_vad(model_vad_gaussian, train_dl, optimizer_gaussian, num_epochs, device)
    #
    # Uniform distribution VAD
    train_ds, train_dl, test_ds, test_dl, latents_train, latents_test = setup_data(batch_size, latent_dim, device)
    model_vad_uniform = VariationalAutoDecoder(latent_dim=64, distribution='uniform').to(device)
    optimizer_uniform = optim.Adam(model_vad_uniform.parameters(), lr=learning_rate)
    print("Training VAD with Uniform distribution...")
    train_vad(model_vad_uniform, train_dl, optimizer_uniform, num_epochs, device)

    #5
    gaussian_vad(model_vad_gaussian, optimizer_gaussian, train_dl, test_dl, latent_dim, device, num_epochs, learning_rate, latents_train, latents_test, train_labels)
    uniform_vad(model_vad_uniform, optimizer_uniform, train_dl, test_dl, latent_dim, device,num_epochs, learning_rate, latents_train, latents_test, train_labels)

    #6
    train_ds, train_dl, test_ds, test_dl, latents_train, latents_test = setup_data(batch_size, latent_dim, device)

    # Pick two samples from the test set (images and labels are swapped in your case)
    label_batch, data_batch = next(iter(test_dl))  # Corrected: data_batch should contain the images

    # Print the shapes of data_batch and label_batch to verify
    print(f"data_batch shape: {data_batch.shape}")  # Expected to be [batch_size, 28, 28] now
    print(f"label_batch shape: {label_batch.shape}")  # This should contain the labels, likely [batch_size]

    # Select the first two samples (ensure they're images, not scalars)
    sample1, sample2 = data_batch[0].unsqueeze(0), data_batch[1].unsqueeze(0)  # Pick two images

    # Flatten the images to [1, 784] for processing
    sample1 = sample1.view(-1, 28 * 28).float()  # Now sample1 is [1, 784]
    sample2 = sample2.view(-1, 28 * 28).float()  # Now sample2 is [1, 784]

    interpolate_and_plot_slerp(model_vad_gaussian, sample1, sample2, device, save_file="interpolations_gaussian_slerp.png")
    interpolate_and_plot_slerp(model_vad_uniform, sample1, sample2, device, save_file="interpolations_uniform_slerp.png")

if __name__ == "__main__":
    main()
