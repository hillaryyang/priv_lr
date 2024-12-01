import torch.nn as nn
import numpy as np
import statistics
import torch.optim as optim
from torch.utils.data import DataLoader

from copy import deepcopy
from private import privatize
from sklearn.metrics import r2_score, root_mean_squared_error

# pytorch lr model
class lrmodel(nn.Module):
  def __init__(self, input_dim):
    super(lrmodel, self).__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    return self.linear(x)

# evaluate private mse and r2
def eval_dp_lr(model, optimizer, criterion, data_loader, data, eta = 1e-2, **params):
  # unpack data and hyperparameters
  x_test, y_test = data
  epochs = params['epochs']
  epsilon = params['epsilon']
  norm_clip = params['norm_clip']
  delta = params['delta']
  batch_size = params['batch_size']

  r2_list = []
  rmse_list = []

  converged = False
  trial = 0
  prev_mean = float("inf")

  while not converged: 
    # clone original model and optimizer
    model_copy = deepcopy(model)
    optimizer_copy = optim.SGD(model_copy.parameters(), lr=optimizer.defaults['lr'])
    data_loader_copy = DataLoader(data_loader, batch_size=batch_size, shuffle=True)

    #train 
    priv_model, priv_optimizer, priv_data_loader = privatize(model_copy, optimizer_copy, data_loader_copy, epochs, epsilon, delta, norm_clip)

    # training the model
    for i in range(epochs):
      for _, (x, y) in enumerate(priv_data_loader):
        priv_optimizer.zero_grad()
        # forward pass
        y_pred = priv_model(x)
        loss = criterion(y_pred, y)

        # backward
        loss.backward()
        priv_optimizer.step()
  
    # get predictions
    y_pred = priv_model(x_test).detach().numpy()
    y_np = y_test.detach().numpy()

    # append to list
    r2_list.append(r2_score(y_np, y_pred))
    rmse_list.append(root_mean_squared_error(y_np, y_pred))

    cur_mean = np.mean(rmse_list) 
    
    # check for convergence
    if trial % 50 == 0:
      if abs(cur_mean - prev_mean) < eta:
        converged = True
      else:
        prev_mean = cur_mean

    if trial % 10 == 0:
      print(f"trial: {trial}, RMSE mean: {cur_mean}")

    trial += 1

  r2_stats = [np.mean(r2_list), np.std(r2_list), statistics.median(r2_list)]
  rmse_stats = [np.mean(rmse_list), np.std(rmse_list), statistics.median(rmse_list)]
  
  return r2_stats, rmse_stats
