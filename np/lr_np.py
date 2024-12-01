import torch.nn as nn
import numpy as np
import statistics

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

# pytorch lr model
class lrmodel(nn.Module):
  def __init__(self, input_dim):
    super(lrmodel, self).__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    return self.linear(x)
  
# evaluates ols model
def eval_ols(x, y, n_trials = 100000):
  r2_list = []
  rmse_list = []

  # train
  for i in range(n_trials):
    # sample x and y
    idx = np.random.choice(x.shape[0], int(0.5 * len(x)), replace=False)
    sample_x = x[idx]
    sample_y = y[idx]

    # fit model
    model = LinearRegression()
    model.fit(sample_x, sample_y)

    # evaluate
    pred = model.predict(sample_x)
    y_np = sample_y.numpy()

    # get r2 and rmse values
    r2 = r2_score(y_np, pred)
    rmse = root_mean_squared_error(y_np, pred)
    
    # append to list
    r2_list.append(r2)
    rmse_list.append(rmse)
    
  r2_stats = [np.mean(r2_list), np.std(r2_list), statistics.median(r2_list)]
  rmse_stats = [np.mean(rmse_list), np.std(rmse_list), statistics.median(rmse_list)]
  
  return rmse_stats, r2_stats