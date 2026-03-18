"""
=========
run_pac.py
=========
Run linear regression with a PAC private guarantee (PAC-LR) using
custom anisotropic noise estimation algorithm (private.py) to learn noise, report RMSE/R2 statistics

Usage: python run_pac.py -d [concrete | lenses | auto]
"""

import math
import ssl
import warnings
import sys
sys.path.append('../')

warnings.simplefilter("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

from lr_pac import membership_pac
from data_loader import load_dataset, parse_datasets

PSR = 0.85  # posterior success rate, moderate privacy
MI = PSR * math.log(2 * PSR) + (1 - PSR) * math.log(2 - 2 * PSR)  # Mutual Information (MI) bound derived from PSR, used to calibrate noise

def run(name: str) -> None:
    """Load dataset, learn/apply noise, and print RMSE/R2 statistics"""
    x, y, _ = load_dataset(name)

    print(f"Training PAC-LR for {name} dataset (PSR={PSR}):")
    rmse_stats, r2_stats, _ = membership_pac((x, y), MI) # training/evaluation

    rmse_mean, rmse_std, rmse_med = rmse_stats
    r2_mean, r2_std, r2_med = r2_stats

    print(f"PAC-LR results for {name} dataset (PSR={PSR})")
    print(f"  RMSE mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    for name in parse_datasets("PAC-LR"):
        run(name)
