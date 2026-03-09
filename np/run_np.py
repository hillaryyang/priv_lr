"""
=========
run_np.py
=========
Non-private OLS baseline

Usage:
    python run_np.py                        # run all datasets
    python run_np.py --dataset concrete     # run only concrete
    python run_np.py --dataset lenses       # run only lenses
    python run_np.py --dataset auto         # run only automobiles
"""

import argparse
import sys
sys.path.append('../')

from lr_np import eval_ols
from data_loader import load_dataset, _LOADERS

def run(name: str) -> None:
    """Load dataset, fit OLS, and print RMSE/R2 statistics."""
    x, y, _ = load_dataset(name, norm=True) # load appropriate dataset
    rmse_stats, r2_stats = eval_ols(x, y) # fit OLS and return RMSE/R2 statistics

    # unpack statistics (mean, std, median) for each metric
    rmse_mean, rmse_std, rmse_med = rmse_stats
    r2_mean, r2_std, r2_med = r2_stats

    # print results
    print(f"Non-private OLS for {name} dataset")
    print(f"  RMSE mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="non-private OLS baseline")
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
