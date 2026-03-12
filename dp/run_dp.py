"""
========
run_dp.py
========
Evaluate differentially private stochastic gradient descent with Opacus. 
Trains a PyTorch LinearRegression model with DPSGD until convergence, report RMSE/R2 statistics.

Usage:
    python run_dp.py                        # run all datasets
    python run_dp.py --dataset concrete     # run only concrete
    python run_dp.py --dataset lenses       # run only lenses
    python run_dp.py --dataset auto         # run only automobiles
"""

import ssl
import warnings
warnings.simplefilter("ignore")

import argparse
import sys
sys.path.append('../')

import torch.nn as nn
import torch.optim as optim

from lr_dp import lrmodel, eval_dp_lr, psr_to_epsilon, DELTA
from data_loader import load_dataset, _LOADERS
ssl._create_default_https_context = ssl._create_unverified_context

LEARNING_RATE = 1e-6
PSR = 0.85 # fix at moderate privacy (0.85) for demonstration purposes

# per-dataset tuned hyperparameters for PSR 0.85
# epsilon is derived from PSR at runtime via psr_to_epsilon
PARAMS = {
    "concrete": {"epochs": 30, "norm_clip": 1.5, "batch_size": 8},
    "lenses":   {"epochs": 30, "norm_clip": 2, "batch_size": 8},
    "auto":     {"epochs": 10, "norm_clip": 1, "batch_size": 16},
}

def run(name: str) -> None:
    """Load dataset, train DP-SGD model until convergence, and print RMSE/R2 statistics"""
    x, y, data_loader = load_dataset(name, norm=True)
    params = PARAMS[name]

    # compute epsilon from posterior success rate and delta
    epsilon = psr_to_epsilon(PSR, DELTA) # compute ε from PSR/delta

    # initialize model, optimizer, and loss
    model = lrmodel(x.shape[1])
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # train and evaluate DP-SGD model
    print(f"Training DP-SGD for {name} dataset (PSR={PSR}, epsilon={epsilon:.6f}):")
    r2_stats, rmse_stats, _ = eval_dp_lr(
        model, optimizer, criterion, data_loader, [x, y],
        epochs=params['epochs'], epsilon=epsilon,
        norm_clip=params['norm_clip'], batch_size=params['batch_size']
    )

    # unpack statistics (mean, std, median) for each metric
    rmse_mean, rmse_std, rmse_med = rmse_stats
    r2_mean, r2_std, r2_med = r2_stats

    # print results
    print(f"DP-SGD results for {name} dataset (PSR={PSR}, ε={epsilon:.6f})")
    print(f"  RMSE mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPSGD-LR")
    parser.add_argument(
        "--dataset",
        choices=list(_LOADERS),
        default=None,
        help="Dataset to run (default: all)"
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(_LOADERS)
    for name in datasets:
        run(name)
