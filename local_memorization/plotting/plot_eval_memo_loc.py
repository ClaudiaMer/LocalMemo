import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# Utilities
# ==================================================

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_coverage_results(path):
    return torch.load(path, map_location="cpu")


def get_step(results, step=None):
    if step is None:
        step = results["global_steps"][-1]
    return step, results["checkpoints"][step]


def safe_log1p(x):
    x = to_numpy(x).astype(float)
    return np.log1p(np.maximum(x, 0.0))


def bin_xy(x, y, num_bins=10, quantile_bins=True):
    x = to_numpy(x).astype(float)
    y = to_numpy(y).astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if quantile_bins:
        edges = np.quantile(x, np.linspace(0, 1, num_bins + 1))
    else:
        edges = np.linspace(x.min(), x.max(), num_bins + 1)

    edges[0] -= 1e-12
    edges[-1] += 1e-12

    xs, ys, yerr, counts = [], [], [], []

    for b in range(num_bins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if m.sum() == 0:
            continue

        xs.append(x[m].mean())
        ys.append(y[m].mean())
        yerr.append(y[m].std() / np.sqrt(m.sum()))
        counts.append(m.sum())

    return np.array(xs), np.array(ys), np.array(yerr), np.array(counts)


def ensure_folder(folder):
    os.makedirs(folder, exist_ok=True)



def _classwise_arrays(local, mem, train_labels, r_key, tau_key=None):
    labels = to_numpy(train_labels).astype(int)
    classes = np.unique(labels)

    local_sparsity = to_numpy(local[r_key]).astype(float)
    assigned = to_numpy(mem["assigned_count_per_train"]).astype(float)

    if tau_key is not None:
        tau_data = mem["tau_metrics"][tau_key]
        y = to_numpy(tau_data["memorization_rate_per_train"]).astype(float)
    else:
        nearest_train_index = to_numpy(mem["nearest_train_index"]).astype(int)
        dominance_per_gen = to_numpy(mem["nearest_train_dominance_log_d2_d1"]).astype(float)

        n_train = len(local_sparsity)
        dominance_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen,
            minlength=n_train,
        )
        dominance_count = np.bincount(nearest_train_index, minlength=n_train)
        y = dominance_sum / np.maximum(dominance_count, 1)
        assigned = dominance_count.astype(float)

    rows = []
    for c in classes:
        mask = (labels == c) & (assigned > 0)

        if mask.sum() == 0:
            continue

        rows.append({
            "class": c,
            "local_sparsity_mean": local_sparsity[mask].mean(),
            "local_sparsity_std": local_sparsity[mask].std(),
            "y_mean": y[mask].mean(),
            "y_std": y[mask].std(),
            "num_points": mask.sum(),
        })

    return rows

def _get_train_labels(local):
    if "train_labels" not in local:
        return None
    return to_numpy(local["train_labels"]).astype(int)

# ==================================================
# Main plotting function
# ==================================================

def scatter_local_coverage_tau_sweep(
    results_path,
    save_folder=None,
    step=None,
    knn_k=10,
    density_quantile=0.05,
    tau_values=None,
    alpha=0.35,
    s=10,
    show=True,
):
    """
    Scatter plots without binning.

    Produces:
        1. memorization_rate_tau vs local sparsity
        2. memorization_rate_tau vs local coverage
        3. memorized_count_tau vs local sparsity
        4. continuous dominance vs local sparsity
    """

    results = torch.load(results_path, map_location="cpu")

    if step is None:
        step = results["global_steps"][-1]

    step_results = results["checkpoints"][step]
    local = results["local_metrics"]
    mem = step_results["memorization_metrics"]

    if save_folder is None:
        save_folder = os.path.join(os.path.dirname(results_path), "scatter_tau_plots")

    os.makedirs(save_folder, exist_ok=True)

    r_key = f"r{knn_k}_dist2"

    if r_key not in local:
        raise KeyError(f"{r_key} not found in local_metrics.")


    local_sparsity = to_numpy(local[r_key]).astype(float)

    assigned_count = to_numpy(mem["assigned_count_per_train"]).astype(float)
    train_labels = _get_train_labels(local)

    tau_metrics = mem["tau_metrics"]

    if tau_values is None:
        tau_values = [tau_metrics[k]["tau"] for k in tau_metrics.keys()]

    # --------------------------------------------------
    # Scatter: continuous dominance of generated samples
    # assigned back to training points
    # --------------------------------------------------
    plt.figure(figsize=(5.5, 4.2))
    baseline_mem = results["test_data_baseline_metrics"]

    for mem_, label in zip([mem, baseline_mem],["Generated samples", "Test data baseline"]): 

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

        
        plt.errorbar(
            local_sparsity[mask],
            dominance_mean[mask],
            yerr=dominance_std[mask],
            fmt=".",
            markersize=s,
            alpha=alpha,
            label=label,
        )

        mask = dominance_count == 0 # we plot points with zero assigned samples, they will just have zero mean and std
        # these are simply the samples the model does not generate closely around.


        plt.errorbar(
            local_sparsity[mask],
            dominance_mean[mask],
            yerr=dominance_std[mask],
            fmt=".",
            markersize=s,
            alpha=alpha,
            label=label +" (no assigned samples)",
        )

    plt.xlabel(rf"local sparsity $r_{{{knn_k}}}^2(x_i)$")
    plt.ylabel(r"mean dominance $\log(d_2/d_1)$")
    plt.title(f"Continuous dominance vs local sparsity, step={step}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_folder, f"scatter_dominance_vs_r{knn_k}_step={step}.png")
    plt.savefig(path, dpi=250)
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Class-wise dominance plot, if labels are available
    # --------------------------------------------------

    if train_labels is not None:
        rows = _classwise_arrays(
            local=local,
            mem=mem,
            train_labels=train_labels,
            r_key=r_key,
            tau_key=None,
        )

        if len(rows) > 0:
            xs = np.array([r["local_sparsity_mean"] for r in rows])
            ys = np.array([r["y_mean"] for r in rows])
            xerr = np.array([r["local_sparsity_std"] for r in rows])
            yerr = np.array([r["y_std"] for r in rows])
            cls = np.array([r["class"] for r in rows])

            plt.figure(figsize=(5.5, 4.2))
            plt.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                capsize=3,
                alpha=0.8,
            )

            for x, y, c in zip(xs, ys, cls):
                plt.text(x, y, str(c), fontsize=8, ha="center", va="bottom")

            plt.xlabel(rf"class mean local sparsity $r_{{{knn_k}}}^2$")
            plt.ylabel(r"class mean dominance $\log(d_2/d_1)$")
            plt.title(f"Class-wise dominance vs sparsity, step={step}")
            plt.tight_layout()

            path = os.path.join(
                save_folder,
                f"classwise_dominance_vs_r{knn_k}_step={step}.png",
            )
            plt.savefig(path, dpi=250)

            if show:
                plt.show()
            else:
                plt.close()
    # --------------------------------------------------
    # Class-colored dominance scatter, if labels available
    # --------------------------------------------------

    if train_labels is not None:
        nearest_train_index = to_numpy(mem["nearest_train_index"]).astype(int)
        dominance_per_gen = to_numpy(mem["nearest_train_dominance_log_d2_d1"]).astype(float)

        n_train = len(local_sparsity)

        dominance_sum = np.bincount(
            nearest_train_index,
            weights=dominance_per_gen,
            minlength=n_train,
        )
        dominance_count = np.bincount(nearest_train_index, minlength=n_train)
        dominance_mean = dominance_sum / np.maximum(dominance_count, 1)

        visited = dominance_count > 0
        classes = np.unique(train_labels)

        plt.figure(figsize=(6, 4.5))

        for c in classes:
            mask = visited & (train_labels == c)
            if mask.sum() == 0:
                continue

            plt.scatter(
                local_sparsity[mask],
                dominance_mean[mask],
                s=s,
                alpha=alpha,
                label=str(c),
            )

        plt.xlabel(rf"local sparsity $r_{{{knn_k}}}^2(x_i)$")
        plt.ylabel(r"mean dominance $\log(d_2/d_1)$")
        plt.title(f"Class-colored dominance vs sparsity, step={step}")
        plt.legend(title="class", fontsize=7, markerscale=3, ncol=2)
        plt.tight_layout()

        path = os.path.join(
            save_folder,
            f"class_colored_dominance_vs_r{knn_k}_step={step}.png",
        )
        plt.savefig(path, dpi=250)

        if show:
            plt.show()
        else:
            plt.close()
    # --------------------------------------------------
    # Tau sweep scatter plots
    # --------------------------------------------------

    for tau in tau_values:
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

        # ----------------------------------------------
        # 1. Memorization rate vs sparsity
        # ----------------------------------------------

        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(
            local_sparsity[visited],
            memorization_rate[visited],
            s=s,
            alpha=alpha,
        )
        plt.xlabel(rf"local sparsity $r_{{{knn_k}}}^2(x_i)$")
        plt.ylabel(rf"memorization rate, $\tau={tau_float:.3f}$")
        plt.title(f"Rate vs sparsity, step={step}")
        plt.tight_layout()

        path = os.path.join(
            save_folder,
            f"scatter_mem_rate_vs_r{knn_k}_tau={tau_float:.3f}_step={step}.png",
        )
        plt.savefig(path, dpi=250)
        if show:
            plt.show()
        else:
            plt.close()

        # ----------------------------------------------
        # 4. Class-wise memorization rate vs sparsity
        # ----------------------------------------------

        if train_labels is not None:
            rows = _classwise_arrays(
                local=local,
                mem=mem,
                train_labels=train_labels,
                r_key=r_key,
                tau_key=tau_key,
            )

            if len(rows) > 0:
                xs = np.array([r["local_sparsity_mean"] for r in rows])
                ys = np.array([r["y_mean"] for r in rows])
                xerr = np.array([r["local_sparsity_std"] for r in rows])
                yerr = np.array([r["y_std"] for r in rows])
                cls = np.array([r["class"] for r in rows])

                plt.figure(figsize=(5.5, 4.2))
                plt.errorbar(
                    xs,
                    ys,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="o",
                    capsize=3,
                    alpha=0.8,
                )

                for x, y, c in zip(xs, ys, cls):
                    plt.text(x, y, str(c), fontsize=8, ha="center", va="bottom")

                plt.xlabel(rf"class mean local sparsity $r_{{{knn_k}}}^2$")
                plt.ylabel(rf"class mean memorization rate, $\tau={tau_float:.3f}$")
                plt.title(f"Class-wise rate vs sparsity, step={step}")
                plt.tight_layout()

                path = os.path.join(
                    save_folder,
                    f"classwise_mem_rate_vs_r{knn_k}_tau={tau_float:.3f}_step={step}.png",
                )
                plt.savefig(path, dpi=250)

                if show:
                    plt.show()
                else:
                    plt.close()

        # ----------------------------------------------
        # Class-colored memorization-rate scatter
        # ----------------------------------------------

        if train_labels is not None:
            classes = np.unique(train_labels)

            plt.figure(figsize=(6, 4.5))

            for c in classes:
                mask = visited & (train_labels == c)
                if mask.sum() == 0:
                    continue

                plt.scatter(
                    local_sparsity[mask],
                    memorization_rate[mask],
                    s=s,
                    alpha=alpha,
                    label=str(c),
                )

            plt.xlabel(rf"local sparsity $r_{{{knn_k}}}^2(x_i)$")
            plt.ylabel(rf"memorization rate, $\tau={tau_float:.3f}$")
            plt.title(f"Class-colored rate vs sparsity, step={step}")
            plt.legend(title="class", fontsize=7, markerscale=3, ncol=2)
            plt.tight_layout()

            path = os.path.join(
                save_folder,
                f"class_colored_mem_rate_vs_r{knn_k}_tau={tau_float:.3f}_step={step}.png",
            )
            plt.savefig(path, dpi=250)

            if show:
                plt.show()
            else:
                plt.close()


        # ----------------------------------------------
        # Raw memorized count vs sparsity
        # ----------------------------------------------

        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(
            local_sparsity,
            memorized_count,
            s=s,
            alpha=alpha,
        )
        plt.xlabel(rf"local sparsity $r_{{{knn_k}}}^2(x_i)$")
        plt.ylabel(rf"memorized count, $\tau={tau_float:.3f}$")
        plt.title(f"Raw count vs sparsity, step={step}")
        plt.tight_layout()

        path = os.path.join(
            save_folder,
            f"scatter_mem_count_vs_r{knn_k}_tau={tau_float:.3f}_step={step}.png",
        )
        plt.savefig(path, dpi=250)
        if show:
            plt.show()
        else:
            plt.close()

    print(f"Saved scatter tau-sweep plots to {save_folder}")