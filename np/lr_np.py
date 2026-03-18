"""
========
lr_np.py
========
Implement non-private linear regression with scikit-learn
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

def eval_ols(x: torch.Tensor, y: torch.Tensor, n_trials: int = 10000) -> tuple:
    """
    Evaluate OLS using bootstrap sampling (50%) for fair comparison with PAC/DP,
    return aggregate RMSE/R2 statistics across all trials

    Args:
        x, y: feature matrix and target vector
        n_trials: # of trials (default 10k)

    Returns:
        rmse_stats: [mean, std. dev, median] of RMSE across trials
        r2_stats: [mean, std. dev, median] of R2 across trials
    """
    x_np, y_np = np.array(x), np.array(y)

    # sampling info
    n_samples = x_np.shape[0]
    sample_size = int(0.5 * n_samples)

    # empty lists to store values
    r2_list = []
    rmse_list = []

    # run bootstrap trials, each fitting OLS on an independent subsample
    for _ in range(n_trials):
        idx = np.random.choice(n_samples, sample_size, replace=False)
        sample_x, sample_y = x_np[idx], y_np[idx]

        model = LinearRegression() # initialize LinearRegression object
        pred = model.fit(sample_x, sample_y).predict(sample_x) # fit model and get predictions

        # record R2/RMSE
        r2_list.append(r2_score(sample_y, pred))
        rmse_list.append(root_mean_squared_error(sample_y, pred))

    # aggregate statistics (mean, std, median) across all trials
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]
    r2_stats = [np.mean(r2_list), np.std(r2_list), np.median(r2_list)]

    return rmse_stats, r2_stats