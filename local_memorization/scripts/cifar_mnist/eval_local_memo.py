import os
import torch

from local_memorization.scripts.cifar_mnist.train_diffusion_model import (
    load_data,
    load_class,
    load_test_data,
    model_save_folder,
    NUM_GENERATED
)

from local_memorization.optim.utils import parse_args, filename_from_args
from local_memorization.optim.eval_local_coverage import eval_local_coverage_hypothesis
from local_memorization.plotting.plot_eval_memo_loc import scatter_local_coverage_tau_sweep


if __name__ == "__main__":

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_list = (2, 5, 10, 20, 50)
    radius_quantiles = (0.01, 0.02, 0.05, 0.10)
    tau_values = (0.10, 0.20, 1 / 3, 0.50, 0.70)

    eval_local_coverage_hypothesis(
        args,
        load_data,
        load_class,
        model_save_folder,
        filename_from_args,
        num_generated=NUM_GENERATED,
        device=torch.device("cuda"),
        load_test_data=load_test_data,
        metric="cosine",
        abs_cosine=False,
        compute_class_stats=True,
        only_final=False,
        eval_steps=[args.steps],
        allow_generate_missing=True,
    )
    print("FINISHED EVAL LOOP, NOW PLOTTING")

    run_folder = os.path.join(
        model_save_folder(args),
        filename_from_args(args),
    )

    coverage_folder = os.path.join(
        run_folder,
        "local_coverage_hypothesis",
    )

    results_path = os.path.join(
        coverage_folder,
        f"local_coverage_hypothesis_gen{NUM_GENERATED}.pt",
    )

    print(results_path)

    results = torch.load(results_path, map_location="cpu")
    final_step = results["global_steps"][-1]

    # Main recommended scatter plots.
    plot_folder = os.path.join(
        coverage_folder,
        f"scatter_tau_figures_gen{NUM_GENERATED}",
    )
    os.makedirs(plot_folder, exist_ok=True)


    scatter_local_coverage_tau_sweep(
        results_path=results_path,
        save_folder=plot_folder,
        step=final_step,
        knn_k=10,
        density_quantile=0.05,
        tau_values=tau_values,
        alpha=0.35,
        s=10,
        show=False,
    )

    # Repeat for several kNN sparsity definitions.
    for knn_k in k_list:
        plot_folder_k = os.path.join(
            coverage_folder,
            f"scatter_tau_figures_gen{NUM_GENERATED}_k{knn_k}",
        )
        os.makedirs(plot_folder_k, exist_ok=True)

        scatter_local_coverage_tau_sweep(
            results_path=results_path,
            save_folder=plot_folder_k,
            step=final_step,
            knn_k=knn_k,
            density_quantile=0.05,
            tau_values=tau_values,
            alpha=0.35,
            s=10,
            show=False,
        )
