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
    transforms.ToTensor(),          # (1,32,32)
])

# ----------------------------
# Load datasets
# ----------------------------
cifar = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_cifar)

# ----------------------------
# Extract tensors
# ----------------------------
def dataset_to_tensor(dataset, N):
    loader = DataLoader(dataset, batch_size=N, shuffle=True)
    images, labels = next(iter(loader))
    return images[:N], labels

print("Loading data...")


cifar_data, labels = dataset_to_tensor(cifar, N_SUBSAMPLE)
cifar_flat = cifar_data.view(N_SUBSAMPLE, -1).to(DEVICE)
print(f"CIFAR shape: {cifar_data.shape}")

classes = labels.unique()

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


def remove_diagonal(dists):
    mask = dists > 0
    return dists[mask]

def summarize(name, dists):
    mean = dists.mean().item()
    std = dists.std().item()
    print(f"\n{name}:")
    print(f"  Mean squared distance: {mean:.6f}")
    print(f"  Std  squared distance: {std:.6f}")

for c in classes:
    print(f"\nComputing class {c} pairwise distances...")
    X_class = cifar_flat[labels==c]
    square_dists = remove_diagonal(pairwise_sq_dists(X_class))
    np.save(f"data/class{c}_square_dists.npy", square_dists.cpu().numpy())
    summarize(f"class {c}", square_dists)

