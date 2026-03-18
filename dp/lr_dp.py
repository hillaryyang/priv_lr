"""
Evaluate DPSGD-LR using Opacus

The algorithm:
1. Train fresh SGD model using Opacus privacy (sampling -> gradient clipping -> add noise)
2. Record predictions (RMSE/R2)
3. Check for stablization within convergence threshold (0.01)
4. Return overall performance statistics (RMSE/R2)
"""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader

from private import privatize

DELTA = 1e-3 # fixed delta (probability of privacy breach) used across all experiments

def psr_to_epsilon(psr: float, delta: float) -> float:
    """
    Convert PSR -> epsilon with analytical formula

    Args:
        psr: posterior success rate, probability adversary identifies membership
        delta: DP delta parameter, probability of privacy breach (fixed at 1e-3)

    Returns:
        epsilon: corresponding DP epsilon bound
    """
    return math.log((1 - delta) / (1 - psr) - 1)

class lrmodel(nn.Module):
    """Simple PyTorch LR model for DP eval"""

    def __init__(self, input_dim: int):
        """Initialize model with input_dim (# of input features)"""
        super(lrmodel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: input tensor data
        Returns: predictions
        """
        return self.linear(x)

def eval_dp_lr(
    model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
    data_loader: DataLoader, data: tuple[torch.Tensor, torch.Tensor],
    epochs: int, norm_clip: float, batch_size: int,
    epsilon: float,
    eta: float = 0.01,
) -> tuple:
    """
    Train/evaluate DPSGD-LR via convergence loop (privatize, train, evaluate) 
    until stabilization within eta (0.01), return RMSE/R2 stats

    Args:
        model: PyTorch linear regression model
        optimizer: SGD optimizer
        criterion: loss function
        data_loader: training DataLoader
        data: (x_test, y_test) tensors for evaluation
        epochs, norm_clip, batch_size: training hyperparameters
        epsilon: DP privacy budget
        eta: convergence threshold for RMSE mean (default 0.01)

    Returns:
        rmse_stats: [mean, std, median] of RMSE across trials
        r2_stats: [mean, std, median] of R2 across trials
        rmse_list: per-trial RMSE values
    """
    x_test, y_test = data # unpack data

    # lists to store the per-training stats
    r2_list = []
    rmse_list = []

    prev_mean = None

    for trial in itertools.count():  # run trials until RMSE mean converges within eta (0.01)
        # create fresh model with re-initialized random weights each trial
        model_copy = lrmodel(model.linear.in_features)
        optimizer_copy = optim.SGD(model_copy.parameters(), lr=optimizer.defaults['lr'])
        shuffled_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)

        # call privatize function to apply DP guarantees
        priv_model, priv_optimizer, priv_data_loader = privatize(
            model_copy, optimizer_copy, shuffled_loader, epochs, norm_clip, epsilon, DELTA
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

        # add to lists
        r2_list.append(r2_score(y_np, y_pred))
        rmse_list.append(root_mean_squared_error(y_np, y_pred))

        cur_mean = np.mean(rmse_list) # calculate current mean

        if trial % 50 == 0: # check convergence every 50 trials
            if prev_mean is not None and abs(cur_mean - prev_mean) < eta:
                break
            prev_mean = cur_mean

        if trial % 10 == 0: # print results every 10 trials
            print(f"Trial: {trial}, Cumulative RMSE mean: {cur_mean:.4f}")

    # aggregate statistics (mean, std, median) across all trials
    r2_stats = [np.mean(r2_list), np.std(r2_list), np.median(r2_list)]
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]

    return rmse_stats, r2_stats, rmse_list