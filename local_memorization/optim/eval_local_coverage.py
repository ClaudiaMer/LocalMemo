# eval_local_coverage.py

import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from torchvision.utils import save_image

from local_memorization.optim.utils import (
    subsample_test_data, build_diffusion_from_args, sample_from_diffusion
)


# ==================================================
# Basic utilities
# ==================================================

def flatten(x):
    return x.view(x.shape[0], -1)


def normalize_flat(x):
    return F.normalize(flatten(x), dim=1)


def metric_flatten(x, metric="cosine"):
    """Flatten tensors and apply the same normalization convention as the metric."""
    if metric == "cosine":
        return normalize_flat(x)
    if metric == "l2":
        return flatten(x)
    raise ValueError(f"Unknown metric: {metric}")


def _safe_cpu_float(x):
    return x.detach().cpu().float()


def compute_similarity(samples_flat, ref_flat, metric="cosine", abs_cosine=True):
    """compute sample similarity

    Args:
        samples_flat (torch.Tensor): samples in shape (N, flat_dim)
        ref_flat (torch.Tensor): reference dataset with shape (N_ref, flat_dim)
        metric (str, optional): Cosine or negative l2 distance. Defaults to "cosine".
        abs_cosine (bool, optional): If true, use |cosine_sim| else plain cosine_sim. Defaults to True.

    Raises:
        ValueError: metric unknown  

    Returns:
        torch.Tensor: similarities of shape (N, N_ref)
    """
    if metric == "cosine":
        sim = samples_flat @ ref_flat.T
        return sim.abs() if abs_cosine else sim

    elif metric == "l2":
        x2 = (samples_flat ** 2).sum(dim=1, keepdim=True)
        y2 = (ref_flat ** 2).sum(dim=1, keepdim=True).T
        dist2 = x2 + y2 - 2 * samples_flat @ ref_flat.T
        dim = samples_flat.shape[1]
        return -dist2/dim

    else:
        raise ValueError(f"Unknown metric: {metric}")


def topk_neighbors(
    query_flat,
    ref_flat,
    k=1,
    metric="cosine",
    abs_cosine=True,
    chunk_size=2048,
):
    """Find closest k neighbors of query vectors in reference vectors

    Args:
        query_flat (torch.Tensor): query vectors, shape (N, flat_dim)
        ref_flat (torch.Tensor): reference vectors, shape (N_ref, flat_dim)
        k (int, optional): Number of nearest neighbors. Defaults to 1.
        metric (str, optional): metric to compute similarity=closeness. Defaults to "cosine".
        abs_cosine (bool, optional): If true, use |cosine_sim| else plain cosine_sim. Defaults to True.
        chunk_size (int, optional): splits reference dataset into chunks of this size to avoid overflow. Defaults to 2048.

    Returns:
        tuple(torch.Tensor, torch.Tensor): top similarities of shape (N, k), top reference indices of shape (N, k)
    """
    candidate_scores = []
    candidate_indices = []

    # iterate over chunks of reference vectors
    for start in range(0, ref_flat.shape[0], chunk_size):
        end = min(start + chunk_size, ref_flat.shape[0])
        ref_chunk = ref_flat[start:end]

        scores = compute_similarity(
            query_flat,
            ref_chunk,
            metric=metric,
            abs_cosine=abs_cosine,
        )
        
        kk = min(k, scores.shape[1])
        # find kk closest reference vectors in chunk
        vals, idx = torch.topk(scores, k=kk, dim=1)

        candidate_scores.append(vals)
        candidate_indices.append(idx + start)

    candidate_scores = torch.cat(candidate_scores, dim=1)
    candidate_indices = torch.cat(candidate_indices, dim=1)

    kk = min(k, candidate_scores.shape[1])
    # find k closest reference vectors over all chuncks
    top_scores, order = torch.topk(candidate_scores, k=kk, dim=1)
    top_indices = torch.gather(candidate_indices, dim=1, index=order)

    return top_scores, top_indices


def score_to_distance(score, metric="cosine"):
    # turns a similarity score into a distance.
    if metric == "cosine":
        return 2.0 - 2.0 * score
    elif metric == "l2":
        return -score
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ==================================================
# Local coverage of training points
# ==================================================

def compute_train_local_coverage(
    train_flat,
    train_labels=None,
    k_list=(2, 5, 10, 20, 50),
    metric="cosine",
    abs_cosine=True,
    chunk_size=2048,
):
    """Compute local coverage metrics of training set

    Args:
        train_flat (torch.Tensor): train dataset as flattened vectors of shape (N, flat_dim)
        train_labels (torch.Tensor, optional): labels of training vectors. Defaults to None.
        k_list (tuple, optional): k-NN measures to compute. Defaults to (2, 5, 10, 20, 50).
        metric (str, optional):  metric to compute similarity=closeness. Defaults to "cosine".
        abs_cosine (bool, optional): If true, use |cosine_sim| else plain cosine_sim. Defaults to True.
        chunk_size (int, optional):  splits reference dataset into chunks of this size to avoid overflow. Defaults to 2048.

    Returns:
        dict: dictionary with metrics.
    """
    max_k = max(k_list) + 1 

    # find closest max_k neighbors in training set
    top_scores, top_indices = topk_neighbors(
        query_flat=train_flat,
        ref_flat=train_flat,
        k=max_k,
        metric=metric,
        abs_cosine=abs_cosine,
        chunk_size=chunk_size,
    )

    # mask to remove self-similarity
    n = train_flat.shape[0]
    arange = torch.arange(n, device=train_flat.device).view(-1, 1)
    is_self = top_indices == arange

    cleaned_scores = []
    cleaned_indices = []

    for i in range(n):
        keep = ~is_self[i] # remove self-similarities
        cleaned_scores.append(top_scores[i][keep][:max_k - 1])
        cleaned_indices.append(top_indices[i][keep][:max_k - 1])

    nn_scores = torch.stack(cleaned_scores, dim=0)
    nn_indices = torch.stack(cleaned_indices, dim=0)
    nn_dist2 = score_to_distance(nn_scores, metric=metric)

    out = {
        "train_neighbor_scores": nn_scores.detach().cpu(),
        "train_neighbor_indices": nn_indices.detach().cpu(),
        "train_neighbor_dist2": nn_dist2.detach().cpu(),
        "k_list": torch.tensor(k_list),
    }

    for k in k_list:
        kk = min(k, nn_dist2.shape[1])
        out[f"r{k}_dist2"] = nn_dist2[:, kk - 1].detach().cpu()
        out[f"mean_dist2_k{k}"] = nn_dist2[:, :kk].mean(dim=1).detach().cpu()
        out[f"mean_score_k{k}"] = nn_scores[:, :kk].mean(dim=1).detach().cpu()

    
    if train_labels is not None:
        train_labels_cpu = train_labels.detach().cpu()
        out["train_labels"] = train_labels_cpu

        for k in k_list:
            kk = min(k, nn_indices.shape[1])
            neigh_labels = train_labels[nn_indices[:, :kk]].detach().cpu()
            same_class = neigh_labels == train_labels_cpu.view(-1, 1)
            out[f"same_class_frac_k{k}"] = same_class.float().mean(dim=1)

    return out


# ==================================================
# Memorization metric with tau sweep
# ==================================================

def compute_sample_level_memorization_tau_sweep(
    samples_flat,
    train_flat,
    test_flat=None,
    train_labels=None,
    test_labels=None,
    metric="cosine",
    abs_cosine=True,
    chunk_size=2048,
    tau_values=(1/6, 1/4, 1/3, 1/2, 2/3),
):
    """Given samples, compute to which degree they are memorized using different values of tau for criterion. 
    A sample is memorized if, given d1 and d2 the distances to the closest and second-closest training datapoint, respectively 
        d2/d1 > 1/tau

    Args:
        samples_flat (torch.Tensor): samples as flattened vectors of shape (N_S, flat_dim)
        train_flat (torch.Tensor): train dataset as flattened vectors of shape (N, flat_dim)
        test_flat (torch.Tensor, optional): test dataset as flattened vectors of shape (N_test, flat_dim). Defaults to None.
        train_labels (torch.Tensor, optional): labels of training data  . Defaults to None.
        test_labels (torch.Tensor, optional): labels of test data. Defaults to None.
        metric (str, optional):  metric to compute similarity=closeness. Defaults to "cosine".
        abs_cosine (bool, optional): If true, use |cosine_sim| else plain cosine_sim. Defaults to True.
        chunk_size (int, optional):  splits reference dataset into chunks of this size to avoid overflow. Defaults to 2048.
        tau_values (tuple, optional): tau values for memorization criterion. Defaults to (1/6, 1/4, 1/3, 1/2, 2/3).

    Returns:
        torch.Tensor: result dict
    """
    # find closest neighbors of samples in training set
    train_top_scores, train_top_indices = topk_neighbors(
        query_flat=samples_flat,
        ref_flat=train_flat,
        k=2,
        metric=metric,
        abs_cosine=abs_cosine,
        chunk_size=chunk_size,
    )

    d1 = score_to_distance(train_top_scores[:, 0], metric=metric)
    d2 = score_to_distance(train_top_scores[:, 1], metric=metric)

    eps = 1e-12
    ratio = d1 / (d2 + eps)
    dominance = torch.log((d2 + eps) / (d1 + eps))

    # count how often a given training data point is closest to a sample. 
    nearest_train_index = train_top_indices[:, 0]
    n_train = train_flat.shape[0]
    assigned_count = torch.bincount(
        nearest_train_index,
        minlength=n_train,
    )

    # iterate over tau criteria
    tau_metrics = {}
    for tau in tau_values:
        tau_float = float(tau)
        is_memorized_tau = ratio < tau_float

        memorized_count_tau = torch.bincount(
            nearest_train_index[is_memorized_tau],
            minlength=n_train,
        )

        memorization_rate_tau = memorized_count_tau.float() / (
            assigned_count.float() + eps
        )

        tau_key = f"tau_{tau_float:.4f}"

        tau_metrics[tau_key] = {
            "tau": tau_float,
            "is_memorized": is_memorized_tau.detach().cpu(),
            "memorized_count_per_train": memorized_count_tau.detach().cpu(),
            "memorization_rate_per_train": memorization_rate_tau.detach().cpu(),
            "total_memorized": int(is_memorized_tau.sum().item()),
            "fraction_memorized_generated": float(is_memorized_tau.float().mean().item()),
        }

    result = {
        "nearest_train_index": nearest_train_index.detach().cpu(),
        "nearest_train_top2_indices": train_top_indices.detach().cpu(),
        "nearest_train_top2_scores": train_top_scores.detach().cpu(),
        "nearest_train_d1": d1.detach().cpu(),
        "nearest_train_d2": d2.detach().cpu(),
        "nearest_train_ratio_d1_d2": ratio.detach().cpu(),
        "nearest_train_dominance_log_d2_d1": dominance.detach().cpu(),
        "assigned_count_per_train": assigned_count.detach().cpu(),
        "tau_metrics": tau_metrics,
        "tau_values": torch.tensor([float(t) for t in tau_values]),
    }

    # compute test metrics for comparison
    if test_flat is not None:
        test_top_scores, test_top_indices = topk_neighbors(
            query_flat=samples_flat,
            ref_flat=test_flat,
            k=1,
            metric=metric,
            abs_cosine=abs_cosine,
            chunk_size=chunk_size,
        )

        train_advantage = train_top_scores[:, 0] - test_top_scores[:, 0]

        result.update({
            "nearest_test_index": test_top_indices[:, 0].detach().cpu(),
            "nearest_test_score": test_top_scores[:, 0].detach().cpu(),
            "train_advantage": train_advantage.detach().cpu(),
            "classified_as_train": (train_advantage > 0).detach().cpu(),
        })

    if train_labels is not None:
        result["nearest_train_label"] = train_labels[nearest_train_index].detach().cpu()

    return result


def compute_test_data_baseline(
    test_data,
    train_flat,
    test_flat,
    train_labels=None,
    test_labels=None,
    metric="cosine",
    abs_cosine=True,
    chunk_size=4096,
    tau_values=(1/6, 1/4, 1/3, 1/2, 2/3),
    device=None,
):
    """Compute equivalent of sample memorization, but for test data instead of samples as baseline."""
    if device is None:
        device = train_flat.device

    if metric == "cosine":
        samples_flat = normalize_flat(test_data.to(device))
    elif metric == "l2":
        samples_flat = flatten(test_data.to(device))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return compute_sample_level_memorization_tau_sweep(
        samples_flat=samples_flat,
        train_flat=train_flat,
        test_flat=test_flat,
        train_labels=train_labels,
        test_labels=test_labels,
        metric=metric,
        abs_cosine=abs_cosine,
        chunk_size=chunk_size,
        tau_values=tau_values,
    )


# ==================================================
# Summaries
# ==================================================

def binned_memorization_by_predictor(
    predictor,
    memorized_count,
    assigned_count=None,
    num_bins=10,
):
    predictor = predictor.float()
    memorized_count = memorized_count.float()

    finite = torch.isfinite(predictor)
    predictor = predictor[finite]
    memorized_count = memorized_count[finite]

    if assigned_count is not None:
        assigned_count = assigned_count.float()[finite]

    q = torch.linspace(0, 1, num_bins + 1)
    edges = torch.quantile(predictor, q)
    edges[0] -= 1e-8
    edges[-1] += 1e-8

    rows = []

    for b in range(num_bins):
        mask = (predictor >= edges[b]) & (predictor < edges[b + 1])
        if mask.sum() == 0:
            continue

        row = {
            "bin": b,
            "left": edges[b].item(),
            "right": edges[b + 1].item(),
            "num_points": int(mask.sum().item()),
            "predictor_mean": predictor[mask].mean().item(),
            "memorized_count_mean": memorized_count[mask].mean().item(),
            "memorized_count_sum": memorized_count[mask].sum().item(),
        }

        if assigned_count is not None:
            row["assigned_count_mean"] = assigned_count[mask].mean().item()
            row["assigned_count_sum"] = assigned_count[mask].sum().item()

        rows.append(row)

    return rows


def class_level_summary(
    train_labels,
    local_metrics,
    mem_metrics,
    predictor_key="r10_dist2",
):
    # compute averages of a given quantity for samples of given classes
    labels = train_labels.detach().cpu()
    classes = torch.unique(labels)

    predictor = local_metrics[predictor_key].float()
    memorized = mem_metrics["memorized_count_per_train"].float()
    assigned = mem_metrics["assigned_count_per_train"].float()

    rows = []

    for c in classes:
        mask = labels == c

        rows.append({
            "class": int(c.item()),
            "num_train": int(mask.sum().item()),
            f"{predictor_key}_mean": predictor[mask].mean().item(),
            f"{predictor_key}_median": predictor[mask].median().item(),
            "memorized_count_sum": memorized[mask].sum().item(),
            "memorized_count_mean": memorized[mask].mean().item(),
            "assigned_count_sum": assigned[mask].sum().item(),
            "assigned_count_mean": assigned[mask].mean().item(),
        })

    return rows


# ==================================================
# Sample loading helpers
# ==================================================

def load_samples_for_step(
    global_step,
    num_generated,
    samples_folder,
    generated_samples_dirs=None,
    strict_samples=True,
):
    sample_name = f"samples_step={global_step}.pt"

    search_paths = [
        os.path.join(samples_folder, sample_name),
    ]

    if generated_samples_dirs is not None:
        for d in generated_samples_dirs:
            search_paths.append(os.path.join(d, sample_name))

    for candidate_path in search_paths:
        if not os.path.exists(candidate_path):
            continue

        saved = torch.load(candidate_path, map_location="cpu")

        if isinstance(saved, dict) and "samples" in saved:
            candidate_samples = saved["samples"]
        elif torch.is_tensor(saved):
            candidate_samples = saved
        else:
            print(f"Skipping {candidate_path}: expected Tensor or dict with key 'samples'.")
            continue

        if len(candidate_samples) < num_generated:
            msg = (
                f"Found samples for step={global_step} at {candidate_path}, "
                f"but only {len(candidate_samples)} < num_generated={num_generated}."
            )
            if strict_samples:
                raise RuntimeError(msg)

            print("Warning:", msg, "Using all available samples.")
            return candidate_samples, candidate_path

        return candidate_samples[:num_generated], candidate_path

    return None, None


# ==================================================
# Main pipeline
# ==================================================

def eval_local_coverage_hypothesis(
    args,
    load_data,
    load_class,
    model_save_folder,
    filename_from_args,
    num_generated=4000,
    device=None,
    suffix="",
    load_test_data=None,
    metric="cosine",
    abs_cosine=True,
    k_list=(2, 5, 10, 20, 50),
    chunk_size=4096,
    save_generated=True,
    save_sample_images=True,
    image_nrow=16,
    compute_class_stats=True,
    only_final=True,
    close_margin=0.0,
    tau_values=(1/6, 1/4, 1/3, 1/2, 2/3),
    # Options for checkpoint-free evaluation.
    eval_steps=None,
    samples_only=False,
    allow_generate_missing=True,
    strict_samples=True,
    generated_samples_dirs=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

    # --------------------------------------------------
    # Checkpoint discovery (optional, only needed if generation is allowed)
    # --------------------------------------------------

    ckpt_files = []
    ckpt_steps_available = set()

    if os.path.isdir(folder):
        ckpt_files = [
            f for f in os.listdir(folder)
            if f.endswith(".pt") and f.startswith("step=")
        ]
        ckpt_files = sorted(
            ckpt_files,
            key=lambda s: int(s.replace("step=", "").replace(".pt", ""))
        )
        ckpt_steps_available = {
            int(f.replace("step=", "").replace(".pt", ""))
            for f in ckpt_files
        }

    if eval_steps is None:
        if only_final and hasattr(args, "steps"):
            eval_steps = [int(args.steps)]
        elif len(ckpt_steps_available) > 0:
            eval_steps = sorted(ckpt_steps_available)
        else:
            raise RuntimeError(
                "No checkpoints found and eval_steps=None. "
                "Pass eval_steps=[...] to evaluate from saved samples only."
            )
    else:
        eval_steps = [int(s) for s in eval_steps]

    # --------------------------------------------------
    # Load train/test data and split
    # --------------------------------------------------

    data_tensor = load_data(args)
    if compute_class_stats:
        labels = load_class(args)
    else:
        labels = None

    # Split seed for train/test split reproduction
    if hasattr(args, "split_seed"):
        split_seed = int(args.split_seed)
    elif hasattr(args, "seed"): 
        split_seed = args.seed*13
        print("no split seed is specified, using original seed *13 as specified in")
    elif len(ckpt_files) > 0:
        first_ckpt = torch.load(os.path.join(folder, ckpt_files[0]), map_location="cpu")
        split_seed = int(first_ckpt["split_seed"])
    else:
        raise RuntimeError(
            "Cannot infer split_seed because no checkpoints are available. "
            "set args.split_seed."
        )

    num_data = data_tensor.shape[0]
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(num_data, generator=g)

    train_indices = perm[:args.N]
    test_indices = perm[args.N:]

    train_data = data_tensor[train_indices]
    test_data = data_tensor[test_indices]

    if compute_class_stats:
        train_labels = labels[train_indices]
        train_labels = train_labels.to(device)
        test_labels = labels[test_indices]
        test_labels = test_labels.to(device)
    else:
        train_labels = test_labels = None

    # Balance train/test reference sizes. 
    # important to keep memorization metrics comparable 
    # if #test >> #train, then the test to generated similarity baseline will be artificially low.
    if len(test_data) >= len(train_data): # we can reuse the test data, but we need to subsample it to match the train size
        test_data = test_data[:len(train_data)]
        if compute_class_stats:
            test_labels = test_labels[:len(train_data)]
    else: # we don't have enough test data, so we need load the separate test data we saved 
        if load_test_data is not None:
            test_data_ext, test_labels_ext = load_test_data(args, return_class=True)
            test_data = subsample_test_data(
                test_data_ext,
                len(train_data),
                seed=split_seed * 17,
            )
            
            if compute_class_stats:
                test_labels = subsample_test_data(
                    test_labels_ext,
                    len(train_data),
                    seed=split_seed * 17,
                )
                test_labels = test_labels.to(device)
            else:
                test_labels = None

    # --------------------------------------------------
    # Compute local coverage and baseline in real space
    # --------------------------------------------------

    # This step characterizes the local geometry of the training data in real space,
    # we will use it to test whether memorization depends on local coverage.

    train_flat = metric_flatten(train_data.to(device), metric=metric)
    test_flat = metric_flatten(test_data.to(device), metric=metric)

    print("Computing train local-coverage predictors in real space...")
    local_metrics = compute_train_local_coverage(
        train_flat=train_flat,
        train_labels=train_labels if compute_class_stats else None,
        k_list=k_list,
        metric=metric,
        abs_cosine=abs_cosine,
        chunk_size=chunk_size,
    )
    print("Computing held-out test-data baseline in real space...")
    test_baseline_metrics = compute_test_data_baseline(
        test_data=test_data[:num_generated],
        train_flat=train_flat,
        test_flat=test_flat,
        train_labels=train_labels if compute_class_stats else None,
        test_labels=test_labels if compute_class_stats else None,
        metric=metric,
        abs_cosine=abs_cosine,
        chunk_size=chunk_size,
        tau_values=tau_values,
        device=device,
    )
    
    # make output folders and save results thus far

    out_folder = os.path.join(folder, "local_coverage_hypothesis" + suffix)
    os.makedirs(out_folder, exist_ok=True)

    samples_folder = os.path.join(out_folder, "generated_samples")
    images_folder = os.path.join(out_folder, "sample_images")

    if save_generated:
        os.makedirs(samples_folder, exist_ok=True)

    if save_sample_images:
        os.makedirs(images_folder, exist_ok=True)

    results = {
        "global_steps": [],
        "checkpoints": {},
        "local_metrics": local_metrics,
        "local_metrics_real": local_metrics,
        "test_data_baseline_metrics": test_baseline_metrics,
        "num_generated": num_generated,
        "split_seed": split_seed,
        "metric": metric,
        "abs_cosine": abs_cosine,
        "k_list": tuple(k_list),
        "tau_values": tuple(float(t) for t in tau_values),
        "close_margin": close_margin,
        "compute_class_stats": compute_class_stats,
        "eval_steps": tuple(eval_steps),
        "samples_only": samples_only,
        "allow_generate_missing": allow_generate_missing,
        "generated_samples_dirs": generated_samples_dirs,
        "args": vars(args),
    }

    torch.save(local_metrics, os.path.join(out_folder, "train_local_coverage_metrics.pt"))
    torch.save(test_baseline_metrics, os.path.join(out_folder, "test_data_baseline_metrics.pt"))    

    # --------------------------------------------------
    # Compute memorization metrics for each evaluation step
    # --------------------------------------------------

    for global_step in eval_steps:
        global_step = int(global_step)

        # try to find already generated samples
        samples, loaded_from = load_samples_for_step(
            global_step=global_step,
            num_generated=num_generated,
            samples_folder=samples_folder,
            generated_samples_dirs=generated_samples_dirs,
            strict_samples=strict_samples,
        )

        # no samples found, regenerate if allowed
        if samples is None:
            if samples_only or not allow_generate_missing:
                raise FileNotFoundError(
                    f"No generated samples found for step={global_step}. "
                    f"Checked local/new folder, old folder, and generated_samples_dirs."
                )

            ckpt_path = os.path.join(folder, f"step={global_step}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"No samples found for step={global_step}, and checkpoint is missing: {ckpt_path}"
                )

            print(f"\nGenerating samples for checkpoint step={global_step}")
            diffmodel, model = build_diffusion_from_args(
                args,
                train_data=train_data,
                device=device,
                checkpoint=ckpt_path,
            )
            with torch.no_grad():
                samples = sample_from_diffusion(
                    diffmodel,
                    args,
                    num_samples=num_generated,
                    train_data=train_data,
                ).detach().cpu()

            loaded_from = ckpt_path
            print(f"Generated {len(samples)} samples for step={global_step}")
        else:
            print(f"Loaded samples for step={global_step} from {loaded_from}")

        sample_path = os.path.join(samples_folder, f"samples_step={global_step}.pt")

        if save_generated and loaded_from != sample_path:
            torch.save(
                {
                    "samples": samples.detach().cpu(),
                    "global_step": global_step,
                },
                sample_path,
            )

        if save_sample_images:
            image_path = os.path.join(images_folder, f"samples_step={global_step}.png")
            save_image(
                samples[:min(256, len(samples))].detach().cpu(),
                image_path,
                nrow=image_nrow,
                normalize=True,
            )

        if metric == "cosine":
            samples_flat = normalize_flat(samples.to(device))
        else:
            samples_flat = flatten(samples.to(device))

        mem_metrics = compute_sample_level_memorization_tau_sweep(
            samples_flat=samples_flat,
            train_flat=train_flat,
            test_flat=test_flat,
            train_labels=train_labels if compute_class_stats else None,
            test_labels=test_labels if compute_class_stats else None,
            metric=metric,
            abs_cosine=abs_cosine,
            chunk_size=chunk_size,
            tau_values=tau_values,
        )

        # Tau-aware binned summaries.
        binned = {}
        tau_metrics = mem_metrics["tau_metrics"]

        for tau_key, tau_data in tau_metrics.items():
            binned[tau_key] = {}

            for k in k_list:
                key = f"r{k}_dist2"
                binned[tau_key][key] = binned_memorization_by_predictor(
                    predictor=local_metrics[key],
                    memorized_count=tau_data["memorized_count_per_train"],
                    assigned_count=mem_metrics["assigned_count_per_train"],
                    num_bins=10,
                )

        if compute_class_stats:
            class_summary = {}

            for tau_key, tau_data in tau_metrics.items():
                mem_for_class = {
                    "memorized_count_per_train": tau_data["memorized_count_per_train"],
                    "assigned_count_per_train": mem_metrics["assigned_count_per_train"],
                }

                class_summary[tau_key] = class_level_summary(
                    train_labels=train_labels,
                    local_metrics=local_metrics,
                    mem_metrics=mem_for_class,
                    predictor_key=(
                        "r10_dist2"
                        if "r10_dist2" in local_metrics
                        else f"r{k_list[0]}_dist2"
                    ),
                )
        else:
            class_summary = None

        step_results = {
            "global_step": global_step,
            "sample_path": sample_path if save_generated else loaded_from,
            "loaded_from": loaded_from,
            "memorization_metrics": mem_metrics,
            "binned_memorization": binned,
            "class_summary": class_summary,
        }

        results["global_steps"].append(global_step)
        results["checkpoints"][global_step] = step_results

        step_save_path = os.path.join(
            out_folder,
            f"coverage_metrics_step={global_step}.pt",
        )
        torch.save(step_results, step_save_path)

        target_tau = 1.0 / 3.0
        tau_key_print = min(
            tau_metrics.keys(),
            key=lambda k: abs(float(tau_metrics[k]["tau"]) - target_tau),
        )
        tau_data_print = tau_metrics[tau_key_print]

        total_mem = tau_data_print["memorized_count_per_train"].sum().item()
        total_assigned = mem_metrics["assigned_count_per_train"].sum().item()
        frac_mem = tau_data_print["fraction_memorized_generated"]

        print(
            f"step={global_step}, "
            f"generated={len(samples)}, "
            f"tau={tau_data_print['tau']:.4f}, "
            f"fraction-memorized={frac_mem:.4f}, "
            f"total memorized assignments={total_mem:.0f}, "
            f"total assignments={total_assigned:.0f}"
        )

    save_path = os.path.join(
        out_folder,
        f"local_coverage_hypothesis_gen{num_generated}.pt",
    )

    torch.save(results, save_path)

    print(f"\nSaved local coverage hypothesis results to {save_path}")

    return results