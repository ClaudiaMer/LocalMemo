import os
import torch
import numpy as np 
from local_memorization.plotting import set_nice_params, get_next_panel_label


from local_memorization.optim.utils import parse_args, filename_from_args
from local_memorization.optim.eval_local_coverage import eval_local_coverage_hypothesis
from local_memorization.plotting.plot_eval_memo_loc import scatter_local_coverage_tau_sweep, to_numpy, bin_xy

import matplotlib.pyplot as plt

def get_folder(N, unet_type, steps):
    name ='./trained/' \
            + f'checkpoints_seed1_N{N}/' \
            + f'seed=1__N={N}__steps={steps}_unet_type={unet_type}_batch_size=100_lr=1.00e-04_weight_decay=0.00e+00_adamW=0_cosine_lr=0_mask_time=0' \
            + "local_coverage_hypothesis/"
    
    return name

def get_data(N, unet_type, steps):

    name = get_folder(N, unet_type, steps) + 'local_coverage_hypothesis_gen4000.pt'

    results = torch.load(name, map_location="cpu")
    return results

def get_results_path(N, unet_type, steps):

    name = get_folder(N, unet_type, steps) + 'local_coverage_hypothesis_gen4000.pt'
    return name

Ns = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
unet_types = ["mini", "medium", "maxi"]

if __name__ == "__main__":
    for N in Ns:
        for unet_type in unet_types: 

            if unet_type =="maxi" and N> 1000: 
                steps = 1000000
            else: 
                steps = 100000
                
            gen = 4000
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            k_list = (2, 5, 10, 20, 50)
            radius_quantiles = (0.01, 0.02, 0.05, 0.10)
            tau_values = (0.10, 0.20, 1 / 3, 0.50, 0.70)


            results = get_data(N, unet_type, steps)
            final_step = results["global_steps"][-1]

            # Main recommended scatter plots.
            plot_folder = os.path.join(
                get_folder(N, unet_type, steps),
                f"scatter_tau_figures_gen{gen}",
            )
            os.makedirs(plot_folder, exist_ok=True)

            scatter_local_coverage_tau_sweep(
                results_path=get_results_path(N, unet_type, steps),
                save_folder=plot_folder,
                step=final_step,
                knn_k=10,
                density_quantile=0.05,
                tau_values=tau_values,
                alpha=0.35,
                s=10,
                show=False,
            )


colors = ['#94003a', '#6390c6']
colors2 = ['#934565', '#6390c6']


def plot_local_dominance(ax, results, N, knn_k=10,step=None, metric="mean_k"):

    baseline_mem = results["test_data_baseline_metrics"]
    if step is None:
        step = results["global_steps"][-1]

    mem = results["checkpoints"][step]["memorization_metrics"]
    local = results["local_metrics"]
    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    local_sparsity = to_numpy(local[r_key]).astype(float)


    for mem_, label, c, c2 in zip([mem, baseline_mem],["Generated", "Test data"], colors, colors2): 

        nearest_train_index = to_numpy(mem_["nearest_train_index"]).astype(int)
        dominance_per_gen = to_numpy(mem_["nearest_train_dominance_log_d2_d1"]).astype(float)

        # Average dominance over generated samples assigned to each training point.
        n_train = len(local_sparsity)

        # for each training point, find the samples that are assigned to it, 
        # then compute the mean and std of the dominance ln(d_2/d_1) values for those samples.
        # where d_1 is the distance to the nearest training point, and d_2 is the distance to the second nearest training point.

        dominance_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen,
            minlength=n_train,
        )

        # for each training point, count how many generated samples are assigned to it.
        dominance_count = np.bincount(
            nearest_train_index,
            minlength=n_train,
        )
        # compute the sum of squares for std calculation
        dominance_sq_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen**2,
            minlength=n_train,
        )

        # mean and std of dominance for each training point, with safe division
        dominance_mean = dominance_sum / np.maximum(dominance_count, 1)
        dominance_var = (
            dominance_sq_sum / np.maximum(dominance_count, 1)
            - dominance_mean**2
        )
        dominance_std = np.sqrt(np.maximum(dominance_var, 0))

        mask = dominance_count > 0 # we exclude points with zero assigned samples, they will just have zero mean and std
        # these are simply the samples the model does not generate closely around.

        """
        ax.errorbar(
            local_sparsity[mask],
            dominance_mean[mask],
            yerr=dominance_std[mask],
            fmt=".",
            markersize=1,
            alpha=0.1,
            label=label,
            color=c
        )
        """
        ax.scatter(
            local_sparsity[mask],
            dominance_mean[mask],
            marker=".",
            s=1,
            alpha=0.3,
            color=c2, 
            rasterized=True
        )
        xb, yb, yerr, counts = bin_xy(
            local_sparsity[mask],
            dominance_mean[mask],
            num_bins=10,
            quantile_bins=True,
        )
        ax.errorbar(
            xb,
            yb,
            yerr=yerr,
            marker=".",
            label=label,
            color=c
        )

        """
        mask = dominance_count == 0 # we plot points with zero assigned samples, they will just have zero mean and std
        # these are simply the samples the model does not generate closely around.


        ax.errorbar(
            local_sparsity[mask],
            dominance_mean[mask],
            yerr=dominance_std[mask],
            fmt="o",
            markersize=s,
            alpha=alpha,
            label=label +" (no assigned samples)",
        )
        """

    #ax.set_xlabel(rf"avg. dist to top-{knn_k} NNs ($x_i$)", fontsize=6)
    ax.set_xlabel(r"data sparsity around $x^{\mu}$", fontsize=6)
    ax.set_ylabel(r"$\log(d_2/d_1)$", fontsize=6)
    
    ax.set_title(get_next_panel_label()+f"Loc. dominance, P={N}", fontsize=6)
    

def plot_memo_tau(ax, results, N, tau_values, knn_k, step=None, metric="mean_k", alpha=0.1, bins=10): 

    baseline_mem = results["test_data_baseline_metrics"]
    if step is None:
        step = results["global_steps"][-1]

    mem = results["checkpoints"][step]["memorization_metrics"]
    tau_metrics = mem["tau_metrics"]
    local = results["local_metrics"]
    assigned_count = to_numpy(mem["assigned_count_per_train"]).astype(float)

    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    
    colors_ = ['#94003a', '#af475d', '#c67682', '#dca3aa', '#efd0d4', '#ffffff']
    for i, tau in enumerate(tau_values):
        tau_float = float(tau)
        tau_key = f"tau_{tau_float:.4f}"

        if tau_key not in tau_metrics:
            # robust lookup in case of float formatting differences
            candidates = [
                k for k, v in tau_metrics.items()
                if abs(float(v["tau"]) - tau_float) < 1e-3
            ]
            if len(candidates) == 0:
                print(f"Skipping tau={tau_float}: not found.")
                continue
            tau_key = candidates[0]

        tau_data = tau_metrics[tau_key]

        memorized_count = to_numpy(
            tau_data["memorized_count_per_train"]
        ).astype(float)

        memorization_rate = to_numpy(
            tau_data["memorization_rate_per_train"]
        ).astype(float)

        visited = assigned_count > 0

        local_sparsity = to_numpy(local[r_key]).astype(float)

        # ----------------------------------------------
        # 1. Memorization rate vs sparsity
        # ----------------------------------------------

        if i ==0:
            ax.scatter(
                local_sparsity[visited],
                memorization_rate[visited],
                s=1,
                marker=".", 
                alpha=alpha,
                color = colors_[i], 
                rasterized=True
            )
        xb, yb, yerr, counts = bin_xy(
            local_sparsity[visited],
            memorization_rate[visited],
            num_bins=bins,
            quantile_bins=True,
        )
        ax.errorbar(
            xb,
            yb,
            yerr=yerr,
            marker=".",
            label=r"$\tau$=%2.2f"%tau,
            color = colors_[i]
        )
        ax.set_xlabel(r"data sparsity around $x^{\mu}$", fontsize=6)
        ax.set_ylabel(rf"mem. rate", fontsize=6)
        ax.set_ylim(-0.03, 1.03)


cols_N = ['#6390c6']
def plot_data_sparsity(ax, results, N,knn_k,
                        metric="mean_k", alpha=0.1,
                        bins=10, color="k", log=False): 

    local = results["local_metrics"]
    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    local_sparsity = to_numpy(local[r_key]).astype(float)
    if log: 
        local_sparsity = np.log(local_sparsity)
    bins_ = np.logspace(np.log10(local_sparsity.min()), np.log10(local_sparsity.max()), bins)
    ax.hist(local_sparsity, histtype="step", density=True, bins=bins_, 
            label=f"N={N}", color=color)
    ax.set_xscale("log")
    

def _get_train_labels(local):
    if "train_labels" not in local:
        return None
    return to_numpy(local["train_labels"]).astype(int)
    
class_colors = ['#00429d', '#465db0', '#6d79c4', '#9097d7', '#b3b6eb', '#ffcab9', '#fd9291', '#e75d6f', '#c52a52', '#93003a']

def plot_data_sparsity_per_class(ax, results, N,knn_k,
                        metric="mean_k", alpha=0.1,
                        bins=10, colors=class_colors, log=False, 
                        class_labels=None, classes=None): 

    local = results["local_metrics"]
    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    train_labels = _get_train_labels(local)
    if train_labels is not None: 
        if classes is None:
            classes = np.unique(train_labels)
        n_classes = len(classes)
        skip = len(colors)//n_classes
        colors = colors[::skip]
        if class_labels is None: 
            class_labels = [str(i+1) for i in range(n_classes)]
        for i,c in enumerate(classes): 
            mask = train_labels == c
            local_sparsity = to_numpy(local[r_key][mask]).astype(float)
            if log: 
                local_sparsity = np.log(local_sparsity)
            
            if len(local_sparsity)< 300: 
                num_bins = 5
            elif len(local_sparsity)< 1000:
                num_bins= 7
            else: 
                num_bins = bins
            bins_ = np.logspace(np.log10(local_sparsity.min()), np.log10(local_sparsity.max()), bins)
            ax.hist(local_sparsity, histtype="step", 
                    density=True, bins=bins_, 
                    color=colors[i], label=class_labels[i], 
                    linewidth=2.1)
    ax.set_xscale("log")
            


def plot_memo_tau_per_class(ax, results, N, tau, knn_k, 
                            step=None, metric="mean_k", alpha=0.1, bins=10, 
                            colors=class_colors, class_labels=None, classes=None): 

    baseline_mem = results["test_data_baseline_metrics"]
    if step is None:
        step = results["global_steps"][-1]

    mem = results["checkpoints"][step]["memorization_metrics"]
    tau_metrics = mem["tau_metrics"]
    local = results["local_metrics"]
    assigned_count = to_numpy(mem["assigned_count_per_train"]).astype(float)

    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    train_labels = _get_train_labels(local)

    if train_labels is not None:
        tau_float = float(tau)
        tau_key = f"tau_{tau_float:.4f}"

        if tau_key not in tau_metrics:
            # robust lookup in case of float formatting differences
            candidates = [
                k for k, v in tau_metrics.items()
                if abs(float(v["tau"]) - tau_float) < 1e-8
            ]
            if len(candidates) == 0:
                print(f"Skipping tau={tau_float}: not found.")
            tau_key = candidates[0]

        tau_data = tau_metrics[tau_key]

        memorized_count = to_numpy(
            tau_data["memorized_count_per_train"]
        ).astype(float)

        memorization_rate = to_numpy(
            tau_data["memorization_rate_per_train"]
        ).astype(float)

        visited = assigned_count > 0

        local_sparsity = to_numpy(local[r_key]).astype(float)

        # ----------------------------------------------
        # 1. Memorization rate vs sparsity
        # ----------------------------------------------
        if classes is None:
            classes = np.unique(train_labels)
        n_classes = len(classes)
        skip = len(colors)//n_classes
        colors = colors[::skip]
        if class_labels is None: 
            class_labels = [str(i+1) for i in range(n_classes)]
        for i, c in enumerate(classes): 
            mask = visited & (train_labels == c)
            ax.scatter(
                local_sparsity[mask],
                memorization_rate[mask],
                s=1,
                marker=".", 
                alpha=alpha,
                color = colors[i], 
                rasterized=True
            )
            if len(local_sparsity[mask])< 300: 
                num_bins = 5
            elif len(local_sparsity[mask])< 1000:
                num_bins= 7
            else: 
                num_bins = 15
            xb, yb, yerr, counts = bin_xy(
                local_sparsity[mask],
                memorization_rate[mask],
                num_bins=num_bins,
                quantile_bins=True,
            )
            ax.errorbar(
                xb,
                yb,
                yerr=yerr,
                marker=".",
                label=class_labels[i],
                color = colors[i]
            )
        ax.set_xlabel(r"data sparsity"+"\n"+ r"around $x^{\mu}$", fontsize=6)
        ax.set_ylabel(rf"mem. rate", fontsize=6)


def plot_local_dominance_per_class(ax, results, N, knn_k=10, 
                                   step=None, metric="mean_k", bins=10, 
                                   colors=class_colors, class_labels=None, 
                                   classes=None):

    baseline_mem = results["test_data_baseline_metrics"]
    if step is None:
        step = results["global_steps"][-1]

    mem = results["checkpoints"][step]["memorization_metrics"]
    local = results["local_metrics"]
    if metric == "kth":
        r_key = f"r{knn_k}_dist2" # distance to k-th NN
    if metric == "mean_k": 
        r_key = f"mean_dist2_k{knn_k}" # mean distance to k-th NN

    local_sparsity = to_numpy(local[r_key]).astype(float)

    train_labels = _get_train_labels(local)

    if train_labels is not None:
        nearest_train_index = to_numpy(mem["nearest_train_index"]).astype(int)
        dominance_per_gen = to_numpy(mem["nearest_train_dominance_log_d2_d1"]).astype(float)

        # Average dominance over generated samples assigned to each training point.
        n_train = len(local_sparsity)

        # for each training point, find the samples that are assigned to it, 
        # then compute the mean and std of the dominance ln(d_2/d_1) values for those samples.
        # where d_1 is the distance to the nearest training point, and d_2 is the distance to the second nearest training point.

        dominance_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen,
            minlength=n_train,
        )

        # for each training point, count how many generated samples are assigned to it.
        dominance_count = np.bincount(
            nearest_train_index,
            minlength=n_train,
        )
        # compute the sum of squares for std calculation
        dominance_sq_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen**2,
            minlength=n_train,
        )

        # mean and std of dominance for each training point, with safe division
        dominance_mean = dominance_sum / np.maximum(dominance_count, 1)
        dominance_var = (
            dominance_sq_sum / np.maximum(dominance_count, 1)
            - dominance_mean**2
        )
        dominance_std = np.sqrt(np.maximum(dominance_var, 0))


        if classes is None:
            classes = np.unique(train_labels)
        n_classes = len(classes)
        skip = len(colors)//n_classes
        colors = colors[::skip]
        if class_labels is None: 
            class_labels = [str(i+1) for i in range(n_classes)]

        for i,c in enumerate(classes): 
            mask = (dominance_count > 0) & (train_labels == c)
            # we exclude points with zero assigned samples, they will just have zero mean and std
            # these are simply the samples the model does not generate closely around.

            ax.scatter(
                local_sparsity[mask],
                dominance_mean[mask],
                marker=".",
                s=1,
                alpha=0.1,
                color=colors[i], 
                rasterized=True
            )
            if len(local_sparsity[mask])< 300: 
                num_bins = 5
            elif len(local_sparsity[mask])< 1000:
                num_bins= 7
            else: 
                num_bins = 15
            xb, yb, yerr, counts = bin_xy(
                local_sparsity[mask],
                dominance_mean[mask],
                num_bins=num_bins,
                quantile_bins=True,
            )
            ax.errorbar(
                xb,
                yb,
                yerr=yerr,
                marker=".",
                label=class_labels[i],
                color=colors[i]
            )


    #ax.set_xlabel(rf"avg. dist to top-{knn_k} NNs ($x_i$)", fontsize=6)
    ax.set_xlabel(r"data sparsity"+"\n"+ r"around $x^{\mu}$", fontsize=6)
    ax.set_ylabel(r"$\log(d_2/d_1)$", fontsize=6)
    
    #ax.set_title(get_next_panel_label()+f"Train sample dominance \n vs local sparsity, N={N}", fontsize=6)
    
import os
import torch
import matplotlib.pyplot as plt
from .train_diffusion_model import DATA_PATH

def load_samples(N, unet_type, steps):

    name = get_folder(N, unet_type, steps) + f'generated_samples/samples_step={steps}.pt'

    samples = torch.load(name, map_location="cpu")
    return samples

def load_data(cluster=False): 
    # load training data
    if not cluster:
        data_tensor = torch.load(DATA_PATH+"train.pt")[0]
    else:
        work_dir = os.environ.get("WORK")
        data_tensor = torch.load(work_dir +"/"
                              +DATA_PATH+"train.pt")[0]
    return data_tensor

def load_class(cluster=False): 
    # load training data
    if not cluster:
        class_tensor = torch.load(DATA_PATH+"train.pt")[1]
    else:
        work_dir = os.environ.get("WORK")
        class_tensor = torch.load(work_dir +"/"
                              +DATA_PATH+"train.pt")[1]
    return class_tensor


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch



def plot_samples_with_closest_train(
    ax,
    results,
    samples,
    N,
    step=None,
    n=8,
    seed=0,
    sort_by="lowest_density",
    normalize=True,
    ncols=2,
    ylabel=False,
    title=True,
    balanced_by_closest_label=False,
    k_per_class=None,
    num_classes=None,
    classes = None
):
    """
    Plot generated samples and their closest training examples.

    New options:
        sort_by="closest_label":
            sort selected generated samples by the label of their closest train image.

        balanced_by_closest_label=True:
            select k_per_class generated samples for each closest-train label.
            Then n = k_per_class * num_classes.
    """

    if step is None:
        step = results["global_steps"][-1]

    r = results["checkpoints"][step]["memorization_metrics"]

    split_seed = results["split_seed"]
    data_tensor = load_data()
    class_tensor = load_class()

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(data_tensor.shape[0], generator=g)

    train_indices = perm[:N]
    train_data = data_tensor[train_indices]
    train_labels = class_tensor[train_indices]

    num_samples = samples.shape[0]

    nearest_train_index = r["nearest_train_index"][:num_samples].long()
    closest_labels_all = train_labels[nearest_train_index].long()

    valid_sample_indices = torch.arange(num_samples)

    # --------------------------------------------------
    # Select generated samples
    # --------------------------------------------------
    if balanced_by_closest_label:
        if classes is None:
            if num_classes is None:
                num_classes = int(closest_labels_all.max().item()) + 1
            classes = list(range(num_classes))
        else: 
            num_classes = len(classes)

        if k_per_class is None:
            assert n % num_classes == 0
            k_per_class = n // num_classes
        else:
            n = k_per_class * num_classes

        selected_chunks = []

        for c in classes:
            class_candidates = valid_sample_indices[closest_labels_all == c]

            if len(class_candidates) == 0:
                print(f"Warning: no generated samples with closest train label {c}")
                continue

            if sort_by == "lowest_density":
                scores = r["pooled_density"][:num_samples][class_candidates]
                order = torch.argsort(scores)

            elif sort_by == "highest_density":
                scores = r["pooled_density"][:num_samples][class_candidates]
                order = torch.argsort(scores, descending=True)

            elif sort_by == "highest_train_advantage":
                scores = r["train_advantage"][:num_samples][class_candidates]
                order = torch.argsort(scores, descending=True)

            elif sort_by in ["closest_label", "random"]:
                g = torch.Generator().manual_seed(seed + int(c))
                order = torch.randperm(len(class_candidates), generator=g)

            else:
                raise ValueError(f"Unknown sort_by={sort_by}")

            class_selected = class_candidates[order[:k_per_class]]

            if len(class_selected) < k_per_class:
                print(
                    f"Warning: label {c} only has "
                    f"{len(class_selected)} samples, requested {k_per_class}"
                )

            selected_chunks.append(class_selected)

        selected = torch.cat(selected_chunks, dim=0)

        # Display grouped by closest-train label.
        #selected = selected[torch.argsort(closest_labels_all[selected])]

    else:
        n = min(n, num_samples)

        if sort_by == "lowest_density":
            selected = torch.argsort(r["pooled_density"][:num_samples])[:n]

        elif sort_by == "highest_density":
            selected = torch.argsort(
                r["pooled_density"][:num_samples],
                descending=True,
            )[:n]

        elif sort_by == "highest_train_advantage":
            selected = torch.argsort(
                r["train_advantage"][:num_samples],
                descending=True,
            )[:n]

        elif sort_by == "closest_label":
            selected = torch.argsort(closest_labels_all)[:n]

        elif sort_by == "random":
            g = torch.Generator().manual_seed(seed)
            selected = torch.randperm(num_samples, generator=g)[:n]

        else:
            raise ValueError(f"Unknown sort_by={sort_by}")

    n = len(selected)

    generated = samples[selected]
    closest_train_idx = nearest_train_index[selected]
    closest_train = train_data[closest_train_idx]
    closest_labels = train_labels[closest_train_idx]

    nrows = int(np.ceil(n / ncols))


    fig = ax.figure
    parent_spec = ax.get_subplotspec()
    ax.remove()

    total_rows = 2 * nrows

    subgs = gridspec.GridSpecFromSubplotSpec(
        total_rows,
        ncols,
        subplot_spec=parent_spec,
        wspace=0.02,
        hspace=0.02,
    )

    grid_axes = []
    for row in range(total_rows):
        row_axes = []
        for col in range(ncols):
            a = fig.add_subplot(subgs[row, col])
            if row == 0 and col == 0:
                bbox1 = a.get_position()
            if row == 1 and col == 0:
                bbox2 = a.get_position()
            a.axis("off")
            row_axes.append(a)
        grid_axes.append(row_axes)

    def show_image(axis, img):
        img = img.detach().cpu()

        if img.ndim == 3:
            if img.shape[0] == 1:
                axis.imshow(img.squeeze(0), cmap="gray", interpolation="nearest")
            elif img.shape[0] == 3:
                img = img.permute(1, 2, 0)
                if normalize:
                    img = img.clamp(0, 1)
                axis.imshow(img, interpolation="nearest")
            else:
                axis.imshow(img[0], cmap="gray", interpolation="nearest")
        else:
            axis.imshow(img, cmap="gray", interpolation="nearest")

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        axis.axis("off")

    for idx in range(n):
        row = idx // ncols
        col = idx % ncols

        show_image(grid_axes[row][col], generated[idx])
        show_image(grid_axes[row + nrows][col], closest_train[idx])

    if ylabel:
        label_x = bbox1.x0 - 0.008 

        gen_y = bbox1.y0 + bbox1.height * 0.7
        train_y = bbox2.y0 + bbox2.height * 0.3

        fig.text(
            label_x,
            gen_y,
            "gener-\nated",
            fontsize=6,
            rotation=90,
            va="center",
            ha="right",
        )

        fig.text(
            label_x,
            train_y,
            "closest\ntrain",
            fontsize=6,
            rotation=90,
            va="center",
            ha="right",
            linespacing=0.9,
        )

    if title:
        grid_axes[0][0].set_title(
            get_next_panel_label() + f"$P={N}$",
            fontsize=6,
            loc="left",
            pad=1,
        )