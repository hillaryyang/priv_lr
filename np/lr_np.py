import torch.nn as nn
import numpy as np
import statistics
import torch.optim as optim

from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, root_mean_squared_error

# pytorch lr model
class lrmodel(nn.Module):
  def __init__(self, input_dim):
    super(lrmodel, self).__init__()
    self.linear = nn.Linear(input_dim, 1)

  def forward(self, x):
    return self.linear(x)
  
# evaluates ols model
def eval_ols(x, y, alpha = None, n_trials = 100000):
  r2_list = []
  rmse_list = []

  # train the model
  for i in range(n_trials):
    # sample x and y
    idx = np.random.choice(x.shape[0], int(0.5 * len(x)), replace=False)
    sample_x = x[idx]
    sample_y = y[idx]

    # fit model
    model = LinearRegression()
    # model = Ridge(alpha)
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

    if i % 10000 == 0:
      print(f"trial: {i}, r2 avg: {np.mean(r2_list)}, rmse avg: {np.mean(rmse_list)}")
    
  r2_stats = [np.mean(r2_list), np.std(r2_list), statistics.median(r2_list)]
  rmse_stats = [np.mean(rmse_list), np.std(rmse_list), statistics.median(rmse_list)]
  
  return rmse_stats, r2_stats

# returns the trained sgd model
def train_sgd(model, optimizer, criterion, train_loader, epochs):
  for _ in range(epochs):
    for _, (x, y) in enumerate(train_loader):
      optimizer.zero_grad()
      
      # forward pass
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # backward
      loss.backward()
      optimizer.step()
    
  # returns trained model
  return model

# evaluates the sgd model over multiple trainings
def eval_sgd(model, optimizer, criterion, train_loader, x_test, y_test, epochs, eta = 1e-2):
  # sum is initially 0
  r2_list = []

  converged = False
  trial = 0
  prev_mean = float("inf")

  while not converged:
    # make copies of the model/optimizer for each iteration
    model_copy = deepcopy(model)
    opt_copy = optim.SGD(model_copy.parameters(), lr=optimizer.defaults['lr'])

    # train the model on the copies
    train_sgd(model_copy, opt_copy, criterion, train_loader, epochs)
    
    # get predictions
    y_pred = model_copy(x_test)

    # convert to np arrays 
    pred_np = y_pred.detach().numpy()
    y_np = y_test.detach().numpy()

    # add r2
    r2_val = r2_score(y_np, pred_np)
    r2_list.append(r2_val)
    
    cur_mean = np.mean(r2_list)
    
    if trial % 50 == 0:
      if abs(cur_mean - prev_mean) < eta:
        converged = True
      else:
        prev_mean = cur_mean

    if trial % 20 == 0:
      print(f"Trial: {trial}, r2 so far: {cur_mean}")

    trial += 1

  r2_mean = np.mean(r2_list)
  r2_std = np.std(r2_list)
  r2_med = statistics.median(r2_list)
  
  return r2_mean, r2_std, r2_med