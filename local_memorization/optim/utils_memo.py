import torch
import os
from torch.nn import functional as F
from local_memorization.optim.utils import (
    split_train_test_by_class, split_train_test, subsample_test_data,
    build_diffusion_from_args, sample_from_diffusion,
)


def eval_memorization(
    args,
    load_data,
    model_save_folder,
    filename_from_args,
    num_generated=4000,
    device=None,
    split_train_test_=None, 
    suffix="", 
    load_test_data = None
):
    """
    Evaluate memorization by computing, for each generated sample,
    the maximum cosine similarity to any training / test sample.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Locate run folder
    # --------------------------------------------------
    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

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
        raise RuntimeError("No checkpoints found for memorization eval.")

    ckpt_files = sorted(
        ckpt_files,
        key=lambda s: int(s.replace("step=", "").replace(".pt", ""))
    )

    # --------------------------------------------------
    # Load split_seed from first checkpoint
    # --------------------------------------------------
    first_ckpt = torch.load(
        os.path.join(folder, ckpt_files[0]),
        map_location="cpu"
    )
    split_seed = first_ckpt["split_seed"]

    # --------------------------------------------------
    # Load data + deterministic split
    # --------------------------------------------------
    if split_train_test_ is None:
        data_tensor = load_data(args)
        train_data, test_data, _ = split_train_test(
            data_tensor,
            args.N,
            seed=split_seed
        )
    else:
        train_data, test_data, _ = split_train_test_(split_seed)

    num_test = len(test_data)
    if num_test >= len(train_data): 
        test_data = test_data[:num_test] #make comparison fair
    else: # len(test_data) < len(train_data):
        if load_test_data is None:
            print(f"Warning: test set size {len(test_data)} is smaller than number of training data ={num_test}. "
                f"Using full test set for evaluation.")
            num_test = len(test_data)
        else: 
            test_data = load_test_data(args)
            test_data = subsample_test_data(test_data, len(train_data), seed=split_seed*17)

    # Flatten + normalize
    train_flat = F.normalize(
        train_data.view(train_data.shape[0], -1).to(device),
        dim=1
    )

    test_flat = F.normalize(
        test_data.view(test_data.shape[0], -1).to(device),
        dim=1
    )

    # --------------------------------------------------
    # Build model + diffusion
    # --------------------------------------------------
    diffmodel, model = build_diffusion_from_args(
        args,
        train_data=train_data,
        device=device,
        checkpoint=first_ckpt,
    )

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    global_steps = []
    max_cosine_similarities = []
    max_cosine_similarities_test = []

    # --------------------------------------------------
    # Loop over checkpoints
    # --------------------------------------------------
    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(folder, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            samples = sample_from_diffusion(
                diffmodel,
                args,
                num_samples=num_generated,
                train_data=train_data,
            )
            samples_flat = F.normalize(
                samples.view(num_generated, -1),
                dim=1
            )

            # Chunked similarity to avoid OOM
            max_sim = []
            max_sim_test = []

            chunk_size = 4096
            for i in range(0, train_flat.shape[0], chunk_size):
                chunk = train_flat[i:i + chunk_size]
                sim = abs(samples_flat @ chunk.T)
                max_sim.append(sim.max(dim=1).values)

            for i in range(0, test_flat.shape[0], chunk_size):
                chunk = test_flat[i:i + chunk_size]
                sim = abs(samples_flat @ chunk.T)
                max_sim_test.append(sim.max(dim=1).values)

            max_sim = torch.stack(max_sim).max(dim=0).values
            max_sim_test = torch.stack(max_sim_test).max(dim=0).values

        global_step = int(ckpt_name.replace("step=", "").replace(".pt", ""))

        global_steps.append(global_step)
        max_cosine_similarities.append(max_sim.cpu())
        max_cosine_similarities_test.append(max_sim_test.cpu())

        print(
            f"step {global_step}: "
            f"train mean={max_sim.mean().item():.4f}, "
            f"test mean={max_sim_test.mean().item():.4f}"
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    save_path = os.path.join(folder, "memorization_cosine_similarity"+suffix+".pt")
    results = {
            "global_steps": global_steps,
            "max_cosine_similarities": max_cosine_similarities,
            "max_cosine_similarities_test": max_cosine_similarities_test,
            "num_generated": num_generated,
            "args": vars(args),
        }
    torch.save(
        results,
        save_path
    )

    print(f"Saved memorization results to {save_path}")
    return results



def eval_memorization_with_labels(
    args,
    load_data,
    load_class,
    model_save_folder,
    filename_from_args,
    num_generated=4000,
    device=None,
    split_train_test_=None, 
    suffix="", 
    load_test_data = None
):
    """
    Evaluate memorization by computing, for each generated sample,
    the maximum cosine similarity to any training / test sample.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Locate run folder
    # --------------------------------------------------
    run_name = filename_from_args(args)
    folder = os.path.join(model_save_folder(args), run_name)

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
        raise RuntimeError("No checkpoints found for memorization eval.")

    ckpt_files = sorted(
        ckpt_files,
        key=lambda s: int(s.replace("step=", "").replace(".pt", ""))
    )

    # --------------------------------------------------
    # Load split_seed from first checkpoint
    # --------------------------------------------------
    first_ckpt = torch.load(
        os.path.join(folder, ckpt_files[0]),
        map_location="cpu"
    )
    split_seed = first_ckpt["split_seed"]

    # --------------------------------------------------
    # Load data + deterministic split
    # --------------------------------------------------
    data_tensor = load_data(args)
    labels = load_class(args)
    num_samples = data_tensor.shape[0]
    g = torch.Generator()
    g.manual_seed(split_seed)

    perm = torch.randperm(num_samples, generator=g)

    N_train = args.N
    train_indices = perm[:N_train]
    test_indices = perm[N_train:]

    train_data = data_tensor[train_indices]
    test_data = data_tensor[test_indices]
    train_labels = labels[train_indices].to(device)
    test_labels = labels[test_indices].to(device)
    
    num_test = len(test_data)
    if num_test >= len(train_data): 
        print(f"test set size {len(test_data)} is larger than num_train={len(train_data)}. ")
        test_data = test_data[:len(train_data)] #make comparison fair
        test_labels = test_labels[:len(train_data)]
    else: # len(test_data) < len(train_data):
        if load_test_data is None:
            print(f"Warning: test set size {len(test_data)} is smaller than number of training data ={num_test}. "
                f"Using full test set for evaluation.")
            num_test = len(test_data)
        else: 
            print("using load_test_data to load the test set for evaluation")

            test_data, test_labels = load_test_data(args, return_class=True)
            test_data = subsample_test_data(test_data, len(train_data), seed=split_seed*17).to(device)
            test_labels = torch.Tensor(subsample_test_data(test_labels, len(train_data), seed=split_seed*17)).to(device)
            print(f"test_data has shape {test_data.shape} and test_labels has shape {test_labels.shape} after subsampling")
    # Flatten + normalize
    train_flat = F.normalize(
        train_data.view(train_data.shape[0], -1).to(device),
        dim=1
    )

    test_flat = F.normalize(
        test_data.view(test_data.shape[0], -1).to(device),
        dim=1
    )

    # --------------------------------------------------
    # Build model + diffusion
    # --------------------------------------------------
    diffmodel, model = build_diffusion_from_args(
        args,
        train_data=train_data,
        device=device,
        checkpoint=first_ckpt,
    )

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    global_steps = []
    max_cosine_similarities = []
    max_cosine_similarities_test = []
    max_cosine_sim_labels = []
    max_cosine_sim_labels_test = []

    # --------------------------------------------------
    # Loop over checkpoints
    # --------------------------------------------------
    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(folder, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            samples = sample_from_diffusion(
                diffmodel,
                args,
                num_samples=num_generated,
                train_data=train_data,
            )
            print(f"Generated {len(samples)} samples for checkpoint {ckpt_name}")
            samples_flat = F.normalize(
                samples.view(num_generated, -1),
                dim=1
            )

            # Chunked similarity to avoid OOM
            chunk_size = 4096
            num_samples = samples_flat.shape[0]

            max_sim_train = torch.full((num_samples,), -1e9, device=device)
            max_labels_train = torch.zeros(num_samples, dtype=train_labels.dtype, device=device)

            max_sim_test = torch.full((num_samples,), -1e9, device=device)
            max_labels_test = torch.zeros(num_samples, dtype=test_labels.dtype, device=device)

            # ----------------------------------------
            # TRAIN
            # ----------------------------------------
            for i in range(0, train_flat.shape[0], chunk_size):
                chunk = train_flat[i:i + chunk_size]
                sim = torch.abs(samples_flat @ chunk.T)  # keep abs as requested

                chunk_max, chunk_idx = sim.max(dim=1)

                better = chunk_max > max_sim_train
                max_sim_train[better] = chunk_max[better]

                max_labels_train[better] = train_labels[i:i + chunk_size][chunk_idx[better]]

            # ----------------------------------------
            # TEST
            # ----------------------------------------
            for i in range(0, test_flat.shape[0], chunk_size):
                chunk = test_flat[i:i + chunk_size]
                sim = torch.abs(samples_flat @ chunk.T)  # keep abs

                chunk_max, chunk_idx = sim.max(dim=1)

                better = chunk_max > max_sim_test
                max_sim_test[better] = chunk_max[better]

                max_labels_test[better] = test_labels[i:i + chunk_size][chunk_idx[better]]
            

        global_step = int(ckpt_name.replace("step=", "").replace(".pt", ""))

        global_steps.append(global_step)
        max_cosine_similarities.append(max_sim_train.cpu())
        max_cosine_similarities_test.append(max_sim_test.cpu())
        max_cosine_sim_labels.append(max_labels_train.cpu())
        max_cosine_sim_labels_test.append(max_labels_test.cpu())    

        print(
            f"num_generated={num_generated}, "
            f"step {global_step}: "
            f"train mean={max_sim_train.mean().item():.4f}, "
            f"test mean={max_sim_test.mean().item():.4f}"
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    save_path = os.path.join(folder, "memorization_cosine_similarity"+suffix+".pt")
    results = {
            "global_steps": global_steps,
            "max_cosine_similarities": max_cosine_similarities,
            "max_cosine_similarities_test": max_cosine_similarities_test,
            "max_cosine_sim_labels": max_cosine_sim_labels,
            "max_cosine_sim_labels_test": max_cosine_sim_labels_test,
            "num_generated": num_generated,
            "args": vars(args),
        }
    torch.save(
        results,
        save_path
    )

    print(f"Saved memorization results to {save_path}")
    return results


def eval_memorization_per_class(
    args,
    load_data,
    load_class,
    model_save_folder,
    filename_from_args,
    num_generated=10000,
    device=None,
    N_train=100,
):
    """
    Evaluate memorization by computing, for each generated sample,
    the maximum cosine similarity to any training / test sample.
    """
    data = load_data(args)
    classes = load_class(args)
    unique_classes = torch.unique(classes)

    memorization_per_class_results = {}
    for c in unique_classes:
        split_train_test_ = lambda seed: split_train_test_by_class(
            data,
            classes,
            target_class=c,
            N_train=N_train,
            seed=seed
        )
        memorization_per_class_results[c.item()] = eval_memorization( 
                args,
                lambda _: None,
                model_save_folder,
                filename_from_args,
                num_generated=num_generated,
                device=device,
                split_train_test_ = split_train_test_, 
                suffix=f"_class{c.item()}"
            )
    return memorization_per_class_results
