## Overview 

This repository contains the code accompanying the paper "Local Coverage Governs Memorization in Diffusion Models". 

The code provides:
- diffusion model training on CIFAR-10 and combination of CIFAR-10 and MNIST datasets,
- evaluation of memorization via nearest-neighbor dominance,
- local sparsity metrics,
- class-wise memorization analysis,
- scripts to reproduce the main figures.

## Quickstart guide: 

1) Make sure you have the required packages (list in requirements.txt) installed. 
2) In this folder, run pip install -e .
3) To test if everything works, run `python -m local_memorization.scripts.cifar_color.preprocess_data`
    This will download and preprocess the CIFAR-10 dataset. It will create two folders, preferentially in the WORK folder, or, if none exists, locally: 
        `cifar_10_data/color #holds the raw data`, 
        `cifar_10_splits/color #holds {test,train,val}.pt (data splits) and {test,train,val}_stats.pt (mean and covariance of each split). `
    After preprocessing, run
        `python -m local_memorization.scripts.cifar_color.train_diffusion_model --cluster`
   (add the `--cluster `argument only if are working on a cluster with a $WORK folder)
   This will train a diffusion model on the training split.If successful, this will produce a folder
   `local_memorization/scripts/cifar_color/trained/checkpoints_seed1_N20000/`
   that contains the checkpoints of the trained model. 

## Reproduction

To reproduce the results of the paper, run the scripts in the following order: 

1) `python -m  local_memorization.scripts.cifar_color.preprocess`
2)  To reproduce the result of the paper, navigate to `local_memorization/scripts/cifar_color/train_diffusion_model.py` and set `NUM_GENERATED=4000`. Then run 
    `python -m local_memorization.scripts.cifar_color.train_diffusion_model  --N 10000 --unet_type maxi --steps 100000 --eval_memo --eval_memo_per_class --eval-local-coverage --num_checkpoints 6`
    and vary `N` for different dataset sizes. Depending on your hardware, this will take around 5-10 hours. 
    !Notation: In comparison to the paper, we here use `N` instead of `P` to denote the number of training examples.
    To skip training and only do evaluation, pass `--skip_training` and/or 
3) Use `python -m local_memorization.scripts.cifar_color.train_diffusion_model` to evaluate local coverage proxies and memorization and produce additional plots. Pass the same arguments as in the `train_diffusion_model.py` run to evaluate the corresponding results. This will create a subfolder in `trained/checkpoints../your_run.../local_coverage_hypothesis` with corresponding results. 
4) Use the `local_memorization/scripts/cifar_color/plot_local_mem.ipyno` notebook for visualization. 

Repeat for combination of CIFAR10 and MNIST using the corresponding `cifar_mnist` folder in `local_memorization/scripts`.


## What to find where 

All code is located in `local_memorization` with the following subfolders: 
- `models` contains `Diffusion` model class as well as Unets of different sizes (`maxi`,`medium`,`mini` and `minimini`).
- `optim` contains training and evaluation utilities as well as the full pipeline.
- `utils` contains miscaelleaous utility functions
- `stats` contains utilities for sampling and comparing Gaussian distributions
- `plotting` contains functions to control the matplotlib style and custom colors
- `scripts` holds the preprocessing, training, evaluation and vizualization scripts for specific data sets. 
