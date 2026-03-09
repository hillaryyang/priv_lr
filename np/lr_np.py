"""
========
lr_np.py
========
Non-private OLS evaluation with scikit-learn LinearRegression
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

def eval_ols(x, y, n_trials=10000):
  """
  Evaluate OLS, using bootstrap sampling (50%) for fair comparison with PAC/DP.
  Return aggregate RMSE/R2 statistics across all trials

  Args:
      x: feature matrix
      y: target vector
      n_trials: # of bootstrap trials, default 10k

  Returns:
      rmse_stats: [mean, std, median] of RMSE across trials
      r2_stats: [mean, std, median] of R2 across trials
  """
  x_np = np.array(x)  # convert tensor to numpy
  y_np = np.array(y)

  n_samples = x_np.shape[0]
  sample_size = int(0.5 * n_samples)  # sample half of data

  r2_list = []
  rmse_list = []

  # run bootstrap trials, each fitting OLS on an independent subsample
  for _ in range(n_trials):
    idx = np.random.choice(n_samples, sample_size, replace=False)  # random indices
    sample_x, sample_y = x_np[idx], y_np[idx]

    # fit OLS and get predictions
    model = LinearRegression()
    pred = model.fit(sample_x, sample_y).predict(sample_x)

    # record R2 and RMSE
    r2_list.append(r2_score(sample_y, pred))
    rmse_list.append(root_mean_squared_error(sample_y, pred))

  # aggregate statistics (mean, std, median) across all trials
  rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]
  r2_stats   = [np.mean(r2_list),   np.std(r2_list),   np.median(r2_list)]

  return rmse_stats, r2_stats
