import os
import random
import numpy as np
import torch

def set_seed(seed=1):
    """
    Set random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    # make seeds very different, e.g. if we have seed = 1,2,3,4, ...
    base_seed = 1000
    seed = base_seed + seed * 123

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For extra safety
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[INFO] Random seed set to: {seed}")


# Example usage
if __name__ == "__main__":
    set_seed(1234)