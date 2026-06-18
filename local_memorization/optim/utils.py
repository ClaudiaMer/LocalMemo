import argparse
import json
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from local_memorization.models.diffusion import Diffusion
from local_memorization.models.u_net import \
      UNet, UNetMini, UNetMiniMini, UNetMedium
from local_memorization.utils.general import steps2epochs
from local_memorization.stats.set_seed import set_seed


def _choose_unet(name, T, in_channels, mask_time=False):
    """Choose a UNet architecture by name

    Args:
        name (str): architecture name
        T (int): number of diffusion steps
        in_channels (int): Number of input channels, 1 for greyscale images 3 for rgb
        mask_time (bool, optional): Whether to mask time embeddings (train as a blind diffusion model). Defaults to False.

    Raises:
        ValueError: Unknown UNet type

    Returns:
        torch.nn.sequential: Unet
    """
    name = str(name).lower()
    if name in {"maxi", "full", "large", "unet"}:
        return UNet(T=T, mask_time=mask_time, in_channels=in_channels)
    if name in {"medium", "med"}:
        return UNetMedium(T=T, mask_time=mask_time, in_channels=in_channels)
    if name in {"mini", "small", "unetmini"}:
        return UNetMini(T=T, mask_time=mask_time, in_channels=in_channels)
    if name in {"minimini", "tiny", "unetminimini"}:
        return UNetMiniMini(T=T, mask_time=mask_time, in_channels=in_channels)
    raise ValueError(f"Unknown UNet type {name!r}.")


def build_diffusion_from_args(args, train_data, device, checkpoint=None, dataloader_for_stats=None):
    """
    Build pixel-space diffusion model from args
    """

    dim = torch.prod(torch.tensor(train_data.shape[1:])).item()
    num_channels = train_data.shape[1]
    diffmodel = Diffusion(None, dim=dim, device=device)

    model = _choose_unet(args.unet_type, T=diffmodel.T, in_channels=num_channels, mask_time=args.mask_time).to(device)
    if checkpoint is not None:
        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

    diffmodel.model = model
    return diffmodel, model


def sample_from_diffusion(diffmodel, args, num_samples, train_data, sample_batch=False):
    """Sample decoded images for either pixel diffusion or latent diffusion."""
    num_channels = train_data.shape[1]
    dim_1d = train_data.shape[-1]
    return diffmodel.sample(num_samples, shape=(num_samples, num_channels, dim_1d, dim_1d))

def _as_serializable_history(history):
    return {k: [float(v) for v in vals] for k, vals in (history or {}).items()}


def _tensor_or_none_to_cpu(x):
    if x is None:
        return None
    return x.detach().cpu() if torch.is_tensor(x) else x

def save_checkpoint(model, optimizer, epoch, loss, path, split_seed, extra=None):
    """
    save a checkpoint and related information. 
    """
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'split_seed': split_seed,
    }
    if extra is not None:
        checkpoint_dict.update(extra)
    torch.save(checkpoint_dict, path)

def split_train_test(data_tensor, N_train, seed=0):
    """
    Deterministically split data_tensor into a train set of size N_train
    and a test set with the remaining samples.
    """
    num_samples = data_tensor.shape[0]

    if N_train > num_samples:
        raise ValueError(
            f"N_train={N_train} is larger than dataset size {num_samples}"
        )

    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(num_samples, generator=g)

    train_indices = perm[:N_train]
    test_indices = perm[N_train:]

    train_data = data_tensor[train_indices]
    test_data = data_tensor[test_indices]

    return train_data, test_data, seed

def split_train_test_by_class(data_tensor, labels, target_class, N_train, seed=0):
    """
    Yields the same train/test split as split_train_test but only 
    for samples of the specified class.

    Args:
        data_tensor: (N, ...)
        labels: (N,)
        target_class: class to select
        seed: random seed

    Returns:
        train_data, test_data, seed
    """
    num_samples = data_tensor.shape[0]

    g = torch.Generator()
    g.manual_seed(seed)

    # Global permutation (shared across all classes)
    perm = torch.randperm(num_samples, generator=g)

    train_indices = perm[:N_train]
    test_indices = perm[N_train:]

    # Select only indices of the desired class
    class_mask = (labels == target_class)
    class_mask_permuted = class_mask[perm]

    train_data = data_tensor[train_indices]
    train_data_from_class = train_data[class_mask_permuted[:N_train]]
    test_data = data_tensor[test_indices]
    test_data_from_class = test_data[class_mask_permuted[N_train:]]

    return train_data_from_class, test_data_from_class, seed

def filename_from_args(args):
    """
    Create a deterministic, filesystem-safe filename suffix.

    Keep names short enough for cluster filesystems.
    Full hyperparameters should be saved separately in args.json / checkpoint metadata,
    not encoded into a path long enough to frighten POSIX.
    """

    keys = [
        "seed",
        "N",
        "steps",
        "unet_type",
        "batch_size",
        "lr",
        "weight_decay",
        "adamW",
        "cosine_lr",
        "mask_time",
    ]

    if getattr(args, "add_high_freq_noise", False) or getattr(args, "add_low_freq_noise", False):
        keys += ["min_freq", "noise_level"]

    parts = []

    for key in keys:
        if not hasattr(args, key):
            continue

        val = getattr(args, key)

        if isinstance(val, bool):
            parts.append(f"{key}={int(val)}")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.2e}")
        else:
            parts.append(f"{key}={val}")

    if getattr(args, "ldm", False):
        vae_tag = getattr(args, "vae_tag", "vae")
        latent_channels = getattr(args, "vae_latent_channels", None)
        beta_kl = getattr(args, "vae_beta_kl", None)
        ldm_unet = getattr(args, "ldm_unet", getattr(args, "unet_type", "unet"))

        # Compact but still informative.
        parts += [
            "ldm=1",
            f"vae={vae_tag}",
            f"zc={latent_channels}",
            f"vhc={'-'.join(str(c) for c in getattr(args, 'vae_hidden_channels', []))}",
            f"bkl={beta_kl:.0e}" if isinstance(beta_kl, float) else f"bkl={beta_kl}",
            f"lunet={ldm_unet}",
        ]

    run_name = "_".join(parts)

    # Emergency fallback: keep readable prefix and append stable hash.
    # Useful if someone, purely hypothetically, invents even more flags.
    max_len = 180
    if len(run_name) > max_len:
        import hashlib

        digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()#[:100]
        readable = "_".join(parts[:6])
        run_name = f"{readable}__hash={digest}"

    return run_name


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000,
                         help="training steps")
    parser.add_argument("--N", type=int, default=20000,
                         help="number of samples")
    parser.add_argument("--weight_decay", type=float,
                         default=0.0, help="weight decay")
    parser.add_argument("--num_checkpoints", type=int,
                         default=5, help="number of checkpoints")
    parser.add_argument("--cluster", help="use when running experiment on cluster to load/save data from WORK",
                         action="store_true")
    parser.add_argument("--adamW", help="whether to use adamW",
                         action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--unet_type", default="mini", type=str)
    parser.add_argument("--eval_memo", help="evaluate memorization metrics",
                         action="store_true")
    parser.add_argument("--eval_memo_per_class", help="compute class-resolved memo stats",
                         action="store_true")
    parser.add_argument("--skip_training", help="don't train, load checkpoints",
                         action="store_true")
    parser.add_argument("--plot_only", help="skip train & eval, only plot",
                         action="store_true")
    parser.add_argument("--mask_time",
                         action="store_true", help="mask time embeddings in diffusion models")


    parser.add_argument(
    "--eval-local-coverage",
    dest="eval_local_coverage",
    action="store_true",
    help="Evaluate local coverage / local sparsity hypothesis after training.",
    )

    parser.add_argument(
        "--local-coverage-samples-only",
        dest="local_coverage_samples_only",
        action="store_true",
        help="Only use already-generated local-coverage samples; do not generate missing samples.",
    )

    parser.add_argument(
        "--local-coverage-no-class-stats",
        dest="local_coverage_no_class_stats",
        action="store_true",
        help="Disable class-conditioned local coverage statistics.",
    )

    parser.add_argument(
        "--local-coverage-metric",
        type=str,
        default="l2",
        choices=["cosine", "l2"],
        help="Metric used for local coverage evaluation.",
    )
        
    # cosine learning rate scheduling
    parser.add_argument("--cosine_lr", help="use cosine annealing learning rate scheduling",
                         action="store_true")

    args = parser.parse_args()
    return args


def run_experiment(load_data, model_save_folder):
    
    args = parse_args()
    set_seed(args.seed)

    if args.skip_training or args.plot_only:
        return args

    # Learning Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_timesteps = 1000
    base_learning_rate = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    num_steps = args.steps
    end = torch.log10(torch.tensor([1.0 * args.steps])).item()
    steps_to_print = torch.logspace(0, end, steps=args.num_checkpoints)
    steps_to_print = sorted(set(int(s.item()) for s in steps_to_print))

    print("checkpoints: ", steps_to_print)

    # Load data
    data_tensor = load_data(args)
    train_data, test_data, split_seed = \
          split_train_test(data_tensor, args.N, seed=args.seed*13)


    dataset = torch.utils.data.TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    diffmodel, model = build_diffusion_from_args(
        args,
        train_data=train_data,
        device=device,
        dataloader_for_stats=dataloader,
    )

    # Optimizer
    if args.adamW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        print("Using cosine learning rate schedule.")
    else:
        scheduler = None

    num_epochs = steps2epochs(num_steps, batch_size, train_data.shape[0])
    global_step = 0
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        #print(f"epoch {epoch} out of {num_epochs}")
        model.train()

        for step, batch in enumerate(dataloader):

            # forward pass
            batch = batch[0].to(device)
            loss = diffmodel.loss(batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step() 

            losses.append(loss.detach().item())
            global_step += 1

            # Save checkpoint
            if steps_to_print and global_step >= steps_to_print[0]:
                run_name = filename_from_args(args)
                folder = os.path.join(model_save_folder(args), run_name)
                os.makedirs(folder, exist_ok=True)

                name = f"step={global_step}.pt"
                path = os.path.join(folder, name)

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    loss.item(),
                    path,
                    split_seed,
                )
                steps_to_print.pop(0)
    print("done")
    return args

def subsample_test_data(test_data, num_test, seed=0):
    if num_test > len(test_data):
        print(f"Warning: test set size {len(test_data)} is smaller than num_test={num_test}. "
              f"Using full test set for evaluation.")
        return test_data

    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(len(test_data), generator=g)
    selected_indices = perm[:num_test]
    print(f"Subsampled test data to {num_test} samples with seed {seed}.")
    return test_data[selected_indices]

def eval_test_losses(args, load_data, model_save_folder, num_test=1000, num_train_eval=1000, load_test_data = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args)+"/", run_name)
    ckpt_files = [
        f for f in os.listdir(folder)
        if f.endswith(".pt") and f.startswith("step=")
    ]

    # Sort by numeric step
    ckpt_files = sorted(
        ckpt_files,
        key=lambda s: int(s.replace("step=", "").replace(".pt", ""))
    )

    if len(ckpt_files) == 0:
        raise RuntimeError("No checkpoints found.")

    # --------------------------------------------------
    # Load first checkpoint to get split_seed
    # --------------------------------------------------
    first_ckpt = torch.load(
        os.path.join(folder, ckpt_files[0]),
        map_location="cpu"
    )
    split_seed = first_ckpt["split_seed"]

    # --------------------------------------------------
    # Reload data and reconstruct train/test split
    # --------------------------------------------------
    data_tensor = load_data(args)
    train_data, test_data, _ = split_train_test(
        data_tensor,
        args.N,
        seed=split_seed
    )
    if len(test_data) < num_test:
        if load_test_data is None:
            print(f"Warning: test set size {len(test_data)} is smaller than num_test={num_test}. "
                f"Using full test set for evaluation.")
            num_test = len(test_data)
        else: 
            print("using load_test_data to load the test set for evaluation")
            test_data = load_test_data(args)
            test_data = subsample_test_data(test_data, num_test, seed=split_seed*17)
    else:
        print(f"test set size {len(test_data)} is larger than num_test={num_test}. ")
        # Limit dataset sizes for evaluation
        test_data = test_data[:num_test]
    train_eval_data = train_data[:num_train_eval]

    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False
    )
    train_eval_loader = DataLoader(
        torch.utils.data.TensorDataset(train_eval_data),
        batch_size=args.batch_size,
        shuffle=False
    )

    # --------------------------------------------------
    # Rebuild model + diffusion
    # --------------------------------------------------
    diffmodel, model = build_diffusion_from_args(
        args,
        train_data=train_data,
        device=device,
        checkpoint=first_ckpt,
        dataloader_for_stats=train_eval_loader,
    )

    # --------------------------------------------------
    # Evaluate all checkpoints
    # --------------------------------------------------
    global_steps = []
    test_losses = []
    train_losses = []

    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(folder, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # -------------------
        # Test loss
        # -------------------
        total_loss_test = 0.0
        total_samples_test = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch[0].to(device)
                loss = diffmodel.loss(batch)
                total_loss_test += loss.item() * batch.shape[0]
                total_samples_test += batch.shape[0]
        avg_test_loss = total_loss_test / total_samples_test

        # -------------------
        # Train eval loss
        # -------------------
        total_loss_train = 0.0
        total_samples_train = 0
        with torch.no_grad():
            for batch in train_eval_loader:
                batch = batch[0].to(device)
                loss = diffmodel.loss(batch)
                total_loss_train += loss.item() * batch.shape[0]
                total_samples_train += batch.shape[0]
        avg_train_loss = total_loss_train / total_samples_train

        global_step = int(ckpt_name[5:-3])
        global_steps.append(global_step)
        test_losses.append(avg_test_loss)
        train_losses.append(avg_train_loss)

        print(f"step {global_step}: test loss = {avg_test_loss:.6f}, train loss = {avg_train_loss:.6f}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    results_path = os.path.join(folder, "losses.pt")
    torch.save(
        {
            "global_steps": global_steps,
            "test_losses": test_losses,
            "train_losses": train_losses,
        },
        results_path
    )

    print(f"Saved train/test losses to {results_path}")


