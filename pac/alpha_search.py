"""
================
alpha_search.py
================
Grid search over regularization parameter alpha for PAC-LR.

Usage:
    python alpha_search.py --dataset [lenses | concrete | auto]
"""

import math
import argparse
import sys
sys.path.append('../../')
sys.path.append('../')

from data_loader import gen_lenses, gen_concrete, gen_auto
from lr_pac import membership_pac

DATASETS = {
    "lenses":   (gen_lenses,   "Lenses"),
    "concrete": (gen_concrete, "Concrete"),
    "auto":     (gen_auto,     "Automobile"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    args = parser.parse_args()

    gen_fn, dataset_name = DATASETS[args.dataset]
    x, y, _ = gen_fn()

    post_success = 0.85
    mi = post_success * math.log(2 * post_success) + (1 - post_success) * math.log(2 - 2 * post_success)
    alphas = [2 ** i for i in range(-10, 10)]

    best_rmse = float("inf")
    best_params = None

    print(f"Grid search over alpha for {dataset_name} (PSR={post_success})")

    for alpha in alphas:
        print(f"  alpha={alpha}...")
        _, rmse_stats, _ = membership_pac((x, y), mi, alpha=alpha)
        rmse_mean, rmse_std, rmse_med = rmse_stats

        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_params = {"alpha": alpha, "mean": rmse_mean, "std": rmse_std, "median": rmse_med}

    print(f"\n{dataset_name} | PSR={post_success} | best alpha={best_params['alpha']}")
    print(f"RMSE  mean={best_params['mean']:.4f}  std={best_params['std']:.4f}  median={best_params['median']:.4f}")

if __name__ == "__main__":
    main()
