"""
=========
run_pac.py
=========
Consolidates auto_pac.py, concrete_pac.py, and lenses_pac.py into a single
script with a --dataset flag.

Usage:
    python run_pac.py                        # run all datasets
    python run_pac.py --dataset concrete     # run only concrete
    python run_pac.py --dataset lenses       # run only lenses
    python run_pac.py --dataset auto         # run only automobiles
"""

import math
import ssl
import warnings
import argparse
import sys
sys.path.append('../')

from lr_pac import membership_pac
from data_loader import load_dataset, _LOADERS

ssl._create_default_https_context = ssl._create_unverified_context

PSR = 0.85
MI = PSR * math.log(2 * PSR) + (1 - PSR) * math.log(2 - 2 * PSR)

def run(name: str) -> None:
    x, y, _ = load_dataset(name, norm=True)

    print(f"Training PAC-LR for {name} dataset (PSR={PSR}):")
    r2_stats, rmse_stats, _ = membership_pac([x, y], MI, learn_basis=True)

    rmse_mean, rmse_std, rmse_med = rmse_stats
    r2_mean, r2_std, r2_med = r2_stats

    print(f"PAC-LR results for {name} dataset (PSR={PSR})")
    print(f"  RMSE mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAC-LR")
    parser.add_argument("--dataset", choices=list(_LOADERS), default=None,
                        help="Dataset to run (default: all)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(_LOADERS)
    for name in datasets:
        run(name)
