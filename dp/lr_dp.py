"""
========
lr_dp.py
========
Differentially private SGD model evaluation with Opacus. Trains a PyTorch
linear regression model with DP-SGD until convergence, returns RMSE/R2 statistics.
"""

import math
import itertools
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from copy import deepcopy
from private import privatize
from sklearn.metrics import r2_score, root_mean_squared_error

DELTA = 1e-3  # fixed delta (probability of privacy breach) used across all experiments
CONV_CHECK = 20
LOG_INTERVAL = 10

def psr_to_epsilon(psr: float, delta: float) -> float:
    """
    Convert posterior success rate (PSR) to epsilon using a calculated bound:
        ε = ln((1 - δ) / (1 - PSR) - 1)

    Args:
        psr: posterior success rate, probability adversary identifies membership
        delta: DP delta parameter, probability of privacy breach (fixed at 1e-3)

    Returns:
        epsilon: corresponding DP epsilon bound
    """
    return math.log((1 - delta) / (1 - psr) - 1)

class lrmodel(nn.Module):
    """Simple single-layer PyTorch linear regression model"""

    def __init__(self, input_dim):
        # initialize linear layer with given input dimensionality
        super(lrmodel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # forward pass, apply linear transformation to input x
        return self.linear(x)

def eval_dp_lr(model: nn.Module, optimizer: optim.Optimizer,
               criterion: nn.Module, data_loader: DataLoader,
               data: list, epochs: int, epsilon: float,
               norm_clip: float, batch_size: int, eta: float = 0.01):
    """
    Evaluate DP-SGD via convergence loop (privatize, train, evaluate) until stabilization
    within eta (0.01), return RMSE/R2 stats

    Args:
        model: initialized PyTorch linear model
        optimizer: SGD optimizer bound to model parameters
        criterion: loss function
        data_loader: full dataset for building per-trial DataLoaders
        data: [x_test, y_test] tensors for evaluation
        epochs: number of training epochs per trial
        epsilon: DP epsilon privacy budget
        norm_clip: gradient clipping norm
        batch_size: batch size for per-trial DataLoaders
        eta: convergence threshold on RMSE mean (defaults to 0.01)

    Returns:
        r2_stats: [mean, std, median] of R2 across trials
        rmse_stats: [mean, std, median] of RMSE across trials
        rmse_list: per-trial RMSE values
    """
    # unpack data
    x_test, y_test = data

    # lists to store the per-training stats
    r2_list = []
    rmse_list = []

    prev_mean = float("inf")

    for trial in itertools.count():  # run trials until RMSE mean converges within eta (0.01)
        # create fresh model with re-initialized weights each trial for meaningful variance
        model_copy = deepcopy(model)
        for layer in model_copy.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        optimizer_copy = optim.SGD(model_copy.parameters(), lr=optimizer.defaults['lr'])
        data_loader_copy = DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        
        # call privatize function to apply differential privacy guarantees
        priv_model, priv_optimizer, priv_data_loader = privatize(
            model_copy, optimizer_copy, data_loader_copy, epochs, epsilon, DELTA, norm_clip
        )

        # train privatized model for specified epochs
        for _ in range(epochs):
            for (x, y) in priv_data_loader:
                priv_optimizer.zero_grad()
                # forward pass
                y_pred = priv_model(x)
                loss = criterion(y_pred, y)
                # backward pass
                loss.backward()
                priv_optimizer.step()

        # get predictions and record R2 and RMSE for this trial
        y_pred = priv_model(x_test).detach().numpy()
        y_np = y_test.detach().numpy()
        r2_list.append(r2_score(y_np, y_pred))
        rmse_list.append(root_mean_squared_error(y_np, y_pred))

        cur_mean = np.mean(rmse_list)

        # check convergence every 50 trials
        if trial % CONV_CHECK == 0:
            if abs(cur_mean - prev_mean) < eta:
                break
            prev_mean = cur_mean

        if trial % LOG_INTERVAL == 0:
            print(f"Trial: {trial}, Cumulative RMSE mean: {cur_mean}")

    # aggregate statistics (mean, std, median) across all trials
    r2_stats   = [np.mean(r2_list),   np.std(r2_list),   np.median(r2_list)]
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]

    return r2_stats, rmse_stats, rmse_list