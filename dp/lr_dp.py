"""
========
lr_dp.py
========
Differentially private SGD linear regression (DPSGD-LR) using Opacus.

Privacy is quantified via epsilon (ε) — smaller ε means stronger privacy and less
information disclosure risk. ε is derived analytically from posterior success rate (PSR),
the adversary's probability of correct membership inference, enabling direct comparison
with PAC-LR under identical privacy conditions.

The algorithm:
1. Privatize SGD model using Opacus (Poisson sampling, gradient clipping, isotropic noise)
2. Train privatized model and record predictions (RMSE/R2)
3. Check for convergence within threshold (0.01)
4. Return overall performance statistics (RMSE/R2)
"""

import math
import itertools
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from private import privatize
from sklearn.metrics import r2_score, root_mean_squared_error

DELTA = 1e-3  # fixed delta (probability of privacy breach) used across all experiments

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
    """Single-layer PyTorch linear regression model (input_dim → 1 output)."""

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: number of input features
        """
        super(lrmodel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (N, input_dim)

        Returns:
            predictions of shape (N, 1)
        """
        return self.linear(x)

def eval_dp_lr(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data_loader: DataLoader,
    data: tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    epsilon: float,
    norm_clip: float,
    batch_size: int,
    eta: float = 0.01,
) -> tuple:
    """
    Train/evaluate DPSGD-LR via convergence loop (privatize, train, evaluate) until
    stabilization within eta (0.01), return RMSE/R2 stats.

    Args:
        model: PyTorch linear regression model
        optimizer: SGD optimizer
        criterion: loss function
        data_loader: training DataLoader
        data: (x_test, y_test) tensors for evaluation
        epochs: training epochs per trial
        epsilon: DP privacy budget
        norm_clip: per-sample gradient clipping threshold
        batch_size: training batch size
        eta: convergence threshold on RMSE mean (default 0.01)

    Returns:
        rmse_stats: [mean, std, median] of RMSE across trials
        r2_stats: [mean, std, median] of R2 across trials
        rmse_list: per-trial RMSE values
    """
    # unpack data
    x_test, y_test = data

    # lists to store the per-training stats
    r2_list = []
    rmse_list = []

    prev_mean = None

    for trial in itertools.count():  # run trials until RMSE mean converges within eta (0.01)
        # create fresh model with re-initialized weights each trial for meaningful variance
        model_copy = lrmodel(model.linear.in_features)
        optimizer_copy = optim.SGD(model_copy.parameters(), lr=optimizer.defaults['lr'])
        shuffled_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)

        # call privatize function to apply differential privacy guarantees
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
        r2_list.append(r2_score(y_np, y_pred))
        rmse_list.append(root_mean_squared_error(y_np, y_pred))

        cur_mean = np.mean(rmse_list)

        if trial % 50 == 0: # check convergence every 50 trials
            if prev_mean is not None and abs(cur_mean - prev_mean) < eta:
                break
            prev_mean = cur_mean

        if trial % 10 == 0: # print results every 10 trials
            print(f"Trial: {trial}, Cumulative RMSE mean: {cur_mean:.4f}")

    # aggregate statistics (mean, std, median) across all trials
    r2_stats   = [np.mean(r2_list),   np.std(r2_list),   np.median(r2_list)]
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]

    return rmse_stats, r2_stats, rmse_list