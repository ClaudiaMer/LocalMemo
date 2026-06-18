import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from local_memorization.optim.utils import (
    split_train_test, filename_from_args, subsample_test_data,
    build_diffusion_from_args, sample_from_diffusion,
)


def visualize_nearest_neighbors(
    args,
    load_data,
    model_save_folder,
    filename_from_args,
    checkpoint_step,
    num_examples=25,
    device=None,
    get_class_prediction = None,
    load_test_data = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    ckpt_path = os.path.join(folder, f"step={checkpoint_step}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    split_seed = ckpt["split_seed"]

    # --------------------------------------------------
    # Load training data
    # --------------------------------------------------
    data_tensor = load_data(args)
    train_data, test_data, _ = split_train_test(
        data_tensor, args.N, seed=split_seed
    )
    num_test = len(test_data)
    if num_test >= len(train_data): 
        print(f"test set size {len(test_data)} is larger than num_train={len(train_data)}. ")
        test_data = test_data[:len(train_data)] #make comparison fair
        #test_labels = test_labels[:len(train_data)]
    else: # len(test_data) < len(train_data):
        if load_test_data is None:
            print(f"Warning: test set size {len(test_data)} is smaller than number of training data ={num_test}. "
                f"Using full test set for evaluation.")
            num_test = len(test_data)
        else: 
            print("using load_test_data to load the test set for evaluation")

            test_data = load_test_data(args)
            test_data = subsample_test_data(test_data, len(train_data), seed=split_seed*17).to(device)
            #test_labels = subsample_test_data(test_labels, len(train_data), seed=split_seed*17).to(device)
            print(f"test_data has shape {test_data.shape} after subsampling")

    train_flat = train_data.view(train_data.shape[0], -1).to(device)
    train_flat = F.normalize(train_flat, dim=1)

    test_flat = test_data.view(test_data.shape[0], -1).to(device)
    test_flat = F.normalize(test_flat, dim=1)

    # --------------------------------------------------
    # Build model + diffusion
    # --------------------------------------------------
    diffmodel, model = build_diffusion_from_args(
        args,
        train_data=train_data,
        device=device,
        checkpoint=ckpt,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------
    with torch.no_grad():
        samples = sample_from_diffusion(
            diffmodel,
            args,
            num_samples=num_examples,
            train_data=train_data,
        )

    # The plotting code below should use the channel count of the decoded
    # images, not the latent channels. For pixel diffusion these are the same
    # as the model input channels; for LDMs, samples have already been decoded
    # back to image space by sample_from_diffusion.
    num_channels = samples.shape[1]
    if not get_class_prediction is None:
        pred_classes = get_class_prediction(args, samples, device)
        print(f"Predicted classes for generated samples: {pred_classes}")

    samples_flat = F.normalize(samples.view(num_examples, -1), dim=1)

    sim = samples_flat @ train_flat.T
    nn_indices = sim.argmax(dim=1)

    sim_test = samples_flat @ test_flat.T
    nn_indices_test = sim_test.argmax(dim=1)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(num_examples, 3, figsize=(6, 2*num_examples))

    for i in range(num_examples):
        gen = samples[i].detach().cpu()
        nn = train_data[nn_indices[i]].detach().cpu()
        nn_test = test_data[nn_indices_test[i]].detach().cpu()


        if num_channels == 1:
            gen = gen.squeeze(0).numpy()
            nn = nn.squeeze(0).numpy()
            nn_test = nn_test.squeeze(0).numpy()    
        else:
            gen = gen.permute(1, 2, 0).numpy()
            nn = nn.permute(1, 2, 0).numpy()    
            nn_test = nn_test.permute(1, 2, 0).numpy()

        if num_channels > 1 and min(gen.min(), nn.min(), nn_test.min()) >= 0 and max(gen.max(), nn.max(), nn_test.max()) <= 1:
            # If data is in [0, 1], we can display directly
            pass
        else:       # Otherwise, we normalize each image to [0, 1] for visualization
            gen = (gen - gen.min()) / (gen.max() - gen.min() + 1e-8)
            nn = (nn - nn.min()) / (nn.max() - nn.min() + 1e-8)
            nn_test = (nn_test - nn_test.min()) / (nn_test.max() - nn_test.min() + 1e-8)
        axes[i, 0].imshow(gen)
        axes[i, 0].set_title(f"Generated, pred class={pred_classes[i].item() if not get_class_prediction is None else 'N/A'}", 
                             fontsize=6, loc="left")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(nn)
        axes[i, 1].set_title("Nearest train", fontsize=6, loc="left")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(nn_test)
        axes[i, 2].set_title("Nearest test", fontsize=6, loc="left")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(folder+f"/step{checkpoint_step}_nearest_neighbors.pdf", dpi=100)


def plot_loss_and_memorization_curves(args, model_save_folder, save=True):
    """
    Plot training/test loss and training/test memorization curves for a given experiment.
    Uses the files saved by eval_test_losses and eval_memorization.
    Saves figure to the same folder as the run.
    """

    # --------------------------------------------------
    # Locate run folder
    # --------------------------------------------------
    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

    # --------------------------------------------------
    # Load memorization results
    # --------------------------------------------------
    if args.eval_memo or args.eval_memo_per_class:
        mem_path = os.path.join(folder, "memorization_cosine_similarity.pt")
        if not os.path.exists(mem_path):
            raise FileNotFoundError(f"Memorization results not found at {mem_path}")
        mem = torch.load(mem_path)

        steps = mem["global_steps"]
        if args.eval_memo:
            train_mem = [s.mean().item() for s in mem["max_cosine_similarities"]]
            test_mem = [s.mean().item() for s in mem["max_cosine_similarities_test"]]

        if args.eval_memo_per_class:
            all_classes = torch.cat(mem["max_cosine_sim_labels"]).unique().tolist()
            memo_per_class = {
                "train": {c: [] for c in all_classes},
                "test":  {c: [] for c in all_classes},
            }
            for i in range(len(steps)):
                
                train_sims = mem["max_cosine_similarities"][i]
                test_sims  = mem["max_cosine_similarities_test"][i]

                train_labels = mem["max_cosine_sim_labels"][i]
                test_labels  = mem["max_cosine_sim_labels_test"][i]

                
                for c in all_classes:
                    # TRAIN
                    mask_train = (train_labels == c)
                    memo_per_class["train"][c].append(train_sims[mask_train])

                    # TEST
                    mask_test = (test_labels == c)
                    memo_per_class["test"][c].append(test_sims[mask_test])
            

    # --------------------------------------------------
    # Load test losses
    # --------------------------------------------------
    
    test_loss_path = os.path.join(folder, "losses.pt")
    if not os.path.exists(test_loss_path):
        raise FileNotFoundError(f"Test loss results not found at {test_loss_path}")
    test_res = torch.load(test_loss_path)
    test_loss_steps = test_res["global_steps"]
    test_losses = test_res["test_losses"]
    train_losses = test_res["train_losses"]

    

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    # Losses
    axes[0].plot(test_loss_steps, train_losses, label="Train loss")
    axes[0].plot(test_loss_steps, test_losses, label="Test loss")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True)

    # Memorization
    if args.eval_memo:
        axes[1].scatter(steps, train_mem, label="Train memorization")
        axes[1].scatter(steps, test_mem, label="Test memorization")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Mean max cosine similarity")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True)

    if args.eval_memo_per_class:
        for c in memo_per_class["train"].keys():
            train_means = [s.mean().item() for s in memo_per_class["train"][c]]
            test_means = [s.mean().item() for s in memo_per_class["test"][c]]
            train_stds = [s.std().item() for s in memo_per_class["train"][c]]
            test_stds = [s.std().item() for s in memo_per_class["test"][c]]
            axes[2].errorbar(steps, train_means, yerr=train_stds, label=f"Train memorization (class {c})")
            axes[2].errorbar(steps, test_means, yerr=test_stds, label=f"Test memorization (class {c})")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Mean max cosine similarity")
    axes[2].set_xscale("log")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    # Save or show
    if save:
        out_path = os.path.join(folder, "loss_and_memorization_curves.pdf")
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    num_classes = len(memo_per_class["train"].keys()) if args.eval_memo_per_class else 0
    if args.eval_memo_per_class and num_classes > 0:
        fig, axes = plt.subplots(num_classes, 1, figsize=(8, 4*num_classes), sharex=True)
        for i, c in enumerate(memo_per_class["train"].keys()):
            train_means = [s.mean().item() for s in memo_per_class["train"][c]]
            test_means = [s.mean().item() for s in memo_per_class["test"][c]]
            train_stds = [s.std().item() for s in memo_per_class["train"][c]]
            test_stds = [s.std().item() for s in memo_per_class["test"][c]]
            axes[i].errorbar(steps, train_means, yerr=train_stds, label=f"Train memorization (class {c})")
            axes[i].errorbar(steps, test_means, yerr=test_stds, label=f"Test memorization (class {c})")
            axes[i].set_xlabel("Training step")
            axes[i].set_ylabel("Mean max cosine similarity")
            axes[i].set_xscale("log")
            axes[i].legend()
            axes[i].grid(True)
        plt.tight_layout()
        if save:
            out_path = os.path.join(folder, "memorization_curves_per_class.pdf")
            plt.savefig(out_path)
            print(f"Saved per-class memorization plot to {out_path}")
    plt.close()
