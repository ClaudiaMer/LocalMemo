import os
from local_memorization.optim.utils import run_experiment, eval_test_losses,\
    filename_from_args
from local_memorization.optim.utils_memo import eval_memorization, eval_memorization_with_labels
from local_memorization.optim.plot_results import plot_loss_and_memorization_curves, \
     visualize_nearest_neighbors
from local_memorization.optim.eval_local_coverage import eval_local_coverage_hypothesis

def run_full_pipeline(
    load_data,
    model_save_folder,
    visualize_step=None,
    num_generated=4000,
    load_class=lambda _: None,
    get_class_prediction= None,
    load_test_data = None
):
    """
    Master pipeline:
      1) Train diffusion model
      2) Evaluate test loss
      3) Evaluate memorization
      4) Produce plots

    Designed to be run once on a cluster node.
    """

    # --------------------------------------------------
    # 1) TRAIN
    # --------------------------------------------------
    print("=" * 80)
    print("Starting training")
    print("=" * 80)

    args = run_experiment(
        load_data=load_data,
        model_save_folder=model_save_folder,
    )

    # Resolve run folder once
    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

    # --------------------------------------------------
    # 2) TEST LOSS EVALUATION
    # --------------------------------------------------
    print("=" * 80)
    print("Evaluating test losses")
    print("=" * 80)

    if not args.plot_only:

        eval_test_losses(
            args=args,
            load_data=load_data,
            model_save_folder=model_save_folder,
            load_test_data=load_test_data
        )

        # --------------------------------------------------
        # 3) MEMORIZATION EVALUATION
        # --------------------------------------------------
        print("=" * 80)
        print("Evaluating memorization")
        print("=" * 80)

        if args.eval_memo and not args.eval_memo_per_class:
            eval_memorization(
                args=args,
                load_data=load_data,
                model_save_folder=model_save_folder,
                filename_from_args=filename_from_args,
                num_generated=num_generated,
                load_test_data=load_test_data
            )
        if args.eval_memo_per_class:
            memorization_per_class_results = eval_memorization_with_labels(
                args=args,
                load_data=load_data,
                load_class=load_class,
                model_save_folder=model_save_folder,
                filename_from_args=filename_from_args,
                num_generated=num_generated,
                load_test_data=load_test_data
            )

        if getattr(args, "eval_local_coverage", False):

            eval_local_coverage_hypothesis(
                args=args,
                load_data=load_data,
                load_class=load_class,
                model_save_folder=model_save_folder,
                filename_from_args=filename_from_args,
                num_generated=num_generated,
                load_test_data=load_test_data,
                eval_steps=[args.steps],
                metric=getattr(args, "local_coverage_metric", "l2"),
                compute_class_stats=not getattr(args, "local_coverage_no_class_stats", False),
                samples_only=getattr(args, "local_coverage_samples_only", False),
            )

    # --------------------------------------------------
    # 4) VISUALIZATION
    # --------------------------------------------------
    print("=" * 80)
    print("Generating plots")
    print("=" * 80)

    # ---- Plot memorization curves
    plot_loss_and_memorization_curves(args=args,
                                      model_save_folder=model_save_folder)
    
    # ---- Optional nearest-neighbor visualization
    visualize_nearest_neighbors(
        args=args,
        load_data=load_data,
        model_save_folder=model_save_folder,
        filename_from_args=filename_from_args,
        checkpoint_step=args.steps,
        get_class_prediction=get_class_prediction, 
        load_test_data=load_test_data
    )


    print("=" * 80)
    print("FULL PIPELINE COMPLETE")
    print(f"Results saved in: {folder}")
    print("=" * 80)