"""
=========
run_np.py
=========
Evaluate non-private OLS linear regression using lr_np.py

Usage: python run_np.py -d [concrete | lenses | auto]
"""

import sys
sys.path.append('../')

from lr_np import eval_ols
from data_loader import load_dataset, parse_datasets

def run(name: str) -> None:
    """Load dataset, fit OLS, and print RMSE/R2 statistics"""
    print(f"Running OLS for {name} dataset...")
    
    x, y, _ = load_dataset(name) # load appropriate dataset
    (rmse_mean, rmse_std, rmse_med), (r2_mean, r2_std, r2_med) = eval_ols(x, y) # unpack stats

    # print results
    print(f"Non-private OLS results for {name} dataset:")
    print(f"  RMSE: mean={rmse_mean:.4f}     std. dev={rmse_std:.4f}     median={rmse_med:.4f}")
    print(f"  R2:   mean={r2_mean:.4f}     std. dev={r2_std:.4f}     median={r2_med:.4f}")

if __name__ == "__main__":
    for name in parse_datasets("Non-private OLS baseline"):
        run(name)
