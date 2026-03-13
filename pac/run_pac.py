"""
=========
run_pac.py
=========
Run linear regression with a PAC private guarantee (PAC-LR) 
Utilizes a custom anisotropic noise estimation algorithm (private.py)  
to learn noise, and report RMSE/R2 statistics

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

warnings.simplefilter("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

from lr_pac import membership_pac
from data_loader import load_dataset, _LOADERS

PSR = 0.85  # posterior success rate — adversary's probability of correct membership inference (0.85 = moderate privacy)
MI = PSR * math.log(2 * PSR) + (1 - PSR) * math.log(2 - 2 * PSR)  # Mutual Information (MI) bound derived from PSR, used to calibrate noise

def run(name: str) -> None:
    """Load dataset, learn/apply noise, and print RMSE/R2 statistics"""
    x, y, _ = load_dataset(name, norm=True)

    print(f"Training PAC-LR for {name} dataset (PSR={PSR}):")
    rmse_stats, r2_stats, _ = membership_pac((x, y), MI)

    rmse_mean, rmse_std, rmse_med = rmse_stats
    r2_mean, r2_std, r2_med = r2_stats

    print(f"PAC-LR results for {name} dataset (PSR={PSR})")
    print(f"  RMSE mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAC-LR")
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
