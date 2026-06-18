import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ----------------------------
# Parameters
# ----------------------------
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SUBSAMPLE = 5000   # per class (adjust!)
BATCH_SIZE = 256

# ----------------------------
# Transforms (unify format)
# ----------------------------
transform_cifar = transforms.Compose([
    transforms.ToTensor()
])

# MNIST: make it compatible with CIFAR
transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 1 → 3 channels
])

# ----------------------------
# Load datasets
# ----------------------------
mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_mnist)
cifar = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_cifar)

# ----------------------------
# Extract tensors
# ----------------------------
def dataset_to_tensor(dataset, N):
    loader = DataLoader(dataset, batch_size=N, shuffle=True)
    images, _ = next(iter(loader))
    return images[:N]

print("Loading data...")

mnist_data = dataset_to_tensor(mnist, N_SUBSAMPLE)
cifar_data = dataset_to_tensor(cifar, N_SUBSAMPLE)

print(f"MNIST shape: {mnist_data.shape}")
print(f"CIFAR shape: {cifar_data.shape}")

# Flatten
mnist_flat = mnist_data.view(N_SUBSAMPLE, -1).to(DEVICE)
cifar_flat = cifar_data.view(N_SUBSAMPLE, -1).to(DEVICE)

D = mnist_flat.shape[1]

# ----------------------------
# Pairwise squared distances
# ----------------------------
def pairwise_sq_dists(X, chunk_size=100):
    """
    Computes all pairwise squared distances averaged over pixels.
    Returns a vector of size ~N^2/2
    """
    N = X.shape[0]
    results = []

    for i in range(0, N, chunk_size):
        Xi = X[i:i+chunk_size]

        # compute against all
        dist = (
            (Xi[:, None, :] - X[None, :, :]) ** 2
        ).mean(dim=2)  # average over pixels

        results.append(dist.flatten())

    return torch.cat(results)

print("\nComputing MNIST pairwise distances...")
mnist_dists = pairwise_sq_dists(mnist_flat)

print("Computing CIFAR pairwise distances...")
cifar_dists = pairwise_sq_dists(cifar_flat)

# ----------------------------
# Remove self-distances (zeros)
# ----------------------------
def remove_diagonal(dists, N):
    mask = dists > 0
    return dists[mask]

mnist_dists = remove_diagonal(mnist_dists, N_SUBSAMPLE)
cifar_dists = remove_diagonal(cifar_dists, N_SUBSAMPLE)

np.save("data/mnist_dists.npy", mnist_dists.cpu().numpy())
np.save("data/cifar_dists.npy", cifar_dists.cpu().numpy())


# ----------------------------
# Statistics
# ----------------------------
def summarize(name, dists):
    mean = dists.mean().item()
    std = dists.std().item()
    print(f"\n{name}:")
    print(f"  Mean squared distance: {mean:.6f}")
    print(f"  Std  squared distance: {std:.6f}")

summarize("MNIST", mnist_dists)
summarize("CIFAR", cifar_dists)


print("MNIST standard deviation", torch.std(mnist_flat))
print("CIFar10 standard deviation", torch.std(cifar_flat))