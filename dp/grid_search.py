"""
================
grid_search.py
================
Grid search over hyperparameters (epochs, norm clip, batch size, learning rate) to find
the best hyperparameter combinations for DPSGD-LR on each dataset and privacy level

Usage:
    python grid_search.py --dataset [lenses | concrete | auto]
"""

import ssl
import warnings
warnings.simplefilter("ignore")

import sys
import argparse
sys.path.append('../')
sys.path.append('../../')

import torch.nn as nn
from itertools import product
import torch.optim as optim
from torch.utils.data import DataLoader

from lr_dp import lrmodel, eval_dp_lr, psr_to_epsilon, DELTA
from data_loader import load_dataset, _LOADERS

ssl._create_default_https_context = ssl._create_unverified_context

PSR = 0.85 # moderate privacy for demonstration purposes
epsilon = psr_to_epsilon(PSR, DELTA)

# parameter grid
epochs = [1, 5, 10, 20, 30]
norm_clip = [10 ** i for i in range(0, 8)]
batch_size = [2, 4, 8, 16, 32]
learning_rate = [10 ** i for i in range(-8, 0)]

# enumerate all combinations of the above hyperparameters
combos = list(product(epochs, norm_clip, batch_size, learning_rate))

def run_grid_search(name: str) -> None:
    """
    Run grid search for a single dataset given the name
    Return: Prints results and best paramaters found
    """
    x, y, data_loader = load_dataset(name, norm=True)

    print(f"Grid search for {name} dataset (PSR={PSR}, epsilon={epsilon:.6f})")

    best_rmse = float('inf') # initialize as negative infinity for tracking
    best_params = None # initialize best parameters

    for (epoch, nc, batch, lr) in combos: # iterate over all combinations
        # training loop identical to that in run_dp.py
        model = lrmodel(x.shape[1])
        optimizer = optim.SGD(model.parameters(), lr)
        criterion = nn.MSELoss()

        print(f"Training epochs={epoch}, norm_clip={nc}, batch_size={batch}, learning_rate={lr}")
        _, rmse_stats, _ = eval_dp_lr(
            model, optimizer, criterion, data_loader, [x, y],
            epochs=epoch, epsilon=epsilon, norm_clip=nc, batch_size=batch
        )

        rmse_mean, rmse_std, rmse_med = rmse_stats # store results for this run

        if rmse_mean < best_rmse: # if better, then replace parameters with current 
            best_rmse = rmse_mean
            best_params = {
                "epochs": epoch, "norm_clip": nc,
                "batch_size": batch, "learning_rate": lr,
                "rmse_mean": rmse_mean, "rmse_std": rmse_std, "rmse_med": rmse_med
            }

        print(f"Results for epochs={epoch}, norm_clip={nc}, batch_size={batch}, learning_rate={lr}:")
        print(f"  RMSE mean={rmse_mean}, std={rmse_std}, median={rmse_med}")
        print("------------------------------------------------------------")

    print(f"\nBest parameters for {name} (PSR={PSR}, epsilon={epsilon:.6f}):")
    print(f"  Epochs={best_params['epochs']}, Norm Clip={best_params['norm_clip']}, "
          f"Batch Size={best_params['batch_size']}, Learning Rate={best_params['learning_rate']}")
    print(f"  RMSE mean={best_params['rmse_mean']}, std={best_params['rmse_std']}, "
          f"median={best_params['rmse_med']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP-SGD grid search")
    parser.add_argument(
        "--dataset",
        choices=list(_LOADERS),
        default=None,
        help="Dataset to run (default: all)"
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(_LOADERS)
    for name in datasets:
        run_grid_search(name)
