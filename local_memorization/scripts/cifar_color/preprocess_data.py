import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import os

# ----------------------------
# Parameters
# ----------------------------
work_dir = os.environ.get("WORK")
if work_dir is None: 
    work_dir = "."
DATA_DIR = work_dir+"/cifar10_data/color"
OUTPUT_DIR = work_dir+"/cifar10_splits/color"
BATCH_SIZE = 128
SEED = 42

torch.manual_seed(SEED)

# ----------------------------
# Transform: Convert to tensor
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# ----------------------------
# Download CIFAR-10
# ----------------------------
print("Downloading CIFAR-10 dataset...")
dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

# Combine train + test (total 60,000 images)
full_dataset = torch.utils.data.ConcatDataset([dataset, testset])
print(f"Total dataset size: {len(full_dataset)}")

# ----------------------------
# Split into Train / Val / Test (each 1/3)
# ----------------------------
total_size = len(full_dataset)
split_size = total_size // 3  # 20,000 each
train_size = val_size = split_size
test_size = total_size - train_size - val_size  # In case of rounding issues

train_dataset, val_dataset, final_test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(final_test_dataset)}")

# ----------------------------
# Save datasets to disk
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_dataset(dataset, filename):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))
    torch.save((images, labels), os.path.join(OUTPUT_DIR, filename))
    print(f"Saved {filename} with shape {images.shape}")

save_dataset(train_dataset, "train.pt")
save_dataset(val_dataset, "val.pt")
save_dataset(final_test_dataset, "test.pt")
print("All datasets saved successfully!")

DATA_PATH = OUTPUT_DIR

# ----------------------------
# Load datasets
# ----------------------------
def load_split(name):
    images, labels = torch.load(f"{DATA_PATH}/{name}.pt")
    print(f"{name} split loaded: images {images.shape}, labels {labels.shape}")
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