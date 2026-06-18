import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os

# ----------------------------
# Parameters
# ----------------------------
work_dir = os.environ.get("WORK")
if work_dir is None: 
    work_dir = ""
DATA_DIR = work_dir + "./combined_data_mnist_cifar"
OUTPUT_DIR = work_dir + "./combined_splits_mnist_cifar"
BATCH_SIZE = 128
SEED = 42

torch.manual_seed(SEED)

# ----------------------------
# Transforms
# ----------------------------

# CIFAR: already 3x32x32
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
print("Downloading datasets...")

cifar_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_cifar)
cifar_test  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_cifar)

mnist_train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_mnist)
mnist_test  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_mnist)

# ----------------------------
# Add dataset labels
# ----------------------------
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_id):
        self.dataset = dataset
        self.dataset_id = dataset_id  # 0 = CIFAR, 1 = MNIST

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, (self.dataset_id, y)  # IGNORE original class!

# Wrap datasets
cifar_full = ConcatDataset([
    LabeledDataset(cifar_train, 0),
    LabeledDataset(cifar_test, 0)
])

mnist_full = ConcatDataset([
    LabeledDataset(mnist_train, 1),
    LabeledDataset(mnist_test, 1)
])

# ----------------------------
# Balance dataset sizes (important!)
# ----------------------------
min_size = min(len(cifar_full), len(mnist_full))

cifar_subset, _ = random_split(
    cifar_full, [min_size, len(cifar_full) - min_size],
    generator=torch.Generator().manual_seed(SEED)
)

mnist_subset, _ = random_split(
    mnist_full, [min_size, len(mnist_full) - min_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Combine
full_dataset = ConcatDataset([cifar_subset, mnist_subset])

print(f"Total dataset size: {len(full_dataset)} (balanced)")

# ----------------------------
# Split into Train / Val / Test
# ----------------------------
total_size = len(full_dataset)
split_size = total_size // 3

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [split_size, split_size, total_size - 2*split_size],
    generator=torch.Generator().manual_seed(SEED)
)

# ----------------------------
# Save datasets
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_dataset(dataset, filename):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))
    torch.save((images, labels), os.path.join(OUTPUT_DIR, filename))
    print(f"Saved {filename} with shape {images.shape}")

save_dataset(train_dataset, "train.pt")
save_dataset(val_dataset, "val.pt")
save_dataset(test_dataset, "test.pt")

print("All datasets saved successfully!")

DATA_PATH = OUTPUT_DIR

# ----------------------------
# Load datasets
# ----------------------------
def load_split(name):
    images, labels = torch.load(f"{DATA_PATH}/{name}.pt")
    print(f"{name} split loaded:"\
          + f"images {images.shape},")
    return images, labels

train_images, train_labels = load_split("train")
val_images, val_labels = load_split("val")
test_images, test_labels = load_split("test")

# ----------------------------
# Function to compute mean and covariance
# ----------------------------
def compute_mean_cov(images):
    """
    images: tensor of shape (N, 1, H, W)
    Returns: mean (D,), covariance (D, D) where D = H*W
    """
    N = images.shape[0]
    # Flatten to shape (N, D)
    X = images.view(N, -1)

    # Compute mean
    mean = X.mean(dim=0)

    # Center data
    X_centered = X - mean

    # Covariance = (X^T * X) / (N - 1)
    cov = (X_centered.T @ X_centered) / (N - 1)

    return mean, cov

# ----------------------------
# Compute statistics
# ----------------------------
print("\nComputing statistics...")

train_mean, train_cov = compute_mean_cov(train_images)
val_mean, val_cov = compute_mean_cov(val_images)
test_mean, test_cov = compute_mean_cov(test_images)

# ----------------------------
# Print results
# ----------------------------
print("\n--- Results ---")
print(f"Train Mean shape: {train_mean.shape}, Cov shape: {train_cov.shape}")
print(f"Val Mean shape:   {val_mean.shape}, Cov shape: {val_cov.shape}")
print(f"Test Mean shape:  {test_mean.shape}, Cov shape: {test_cov.shape}")

# Example: first 10 mean values
print("\nFirst 10 mean pixel values (Train):")
print(train_mean[:10])

# ----------------------------
# Save statistics (optional)
# ----------------------------
torch.save({"mean": train_mean, "cov": train_cov}, f"{DATA_PATH}/train_stats.pt")
torch.save({"mean": val_mean, "cov": val_cov}, f"{DATA_PATH}/val_stats.pt")
torch.save({"mean": test_mean, "cov": test_cov}, f"{DATA_PATH}/test_stats.pt")

print("\nStatistics saved successfully!")