"""
=========
lr_pac.py
=========
Evaluate linear regression regression with a PAC membership privacy guarantee (PAC-LR)

Privacy is quantified via posterior success rate (PSR), which is converted to 
a Mutual Information (MI) bound, used to calibrate anisotropic noise levels

The algorithm:
1. Estimate per-dimension noise via membership_privacy() (private.py)
2. Repeatedly resample the training data, fit OLS, inject learned anisotropic
   Gaussian noise into the weights, and evaluate until RMSE converges
3. Return overall performance statistics (RMSE/R2)
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from private import membership_privacy, privatize, get_samples

def run_lr(data: tuple[torch.Tensor, torch.Tensor]) -> tuple:
    """
    Helper OLS model used as mechanism in PAC training

    Args:
        data: (train_x, train_y) tensors

    Returns:
        model: fitted LinearRegression model
        tot_weights: flattened array of model weights
    """
    train_x, train_y = data

    model = LinearRegression()
    model.fit(train_x, np.array(train_y).ravel())  # simple OLS fit using training data

    weight = model.coef_
    intercept = model.intercept_
    tot_weights = np.hstack((weight, intercept)).flatten() # compile weights

    return model, tot_weights

def membership_pac(
    train_data: tuple[torch.Tensor, torch.Tensor],
    mi: float,
    eta: float = 1e-2,
    n_draws: int = 1000,
) -> tuple:
    """
    Train/evaluate PAC-LR until stabilization, return RMSE/R2 stats

    Noise is estimated once via membership_privacy(), then at each trial:
    - resample the full training set
    - fit OLS and perturb weights with the learned noise
    - record RMSE/R2 on the resampled training data

    Args:
        train_data: (train_x, train_y) tensors
        mi: Mutual Information bound used in noise calibration
        eta: convergence threshold on RMSE mean (default 0.01)
        n_draws: noise draws per bootstrap sample (default 1000)

    Returns:
        rmse_stats: [mean, std, median] of RMSE across trials
        r2_stats: [mean, std, median] of R2 across trials
        rmse_list: per-trial RMSE values
    """
    train_x, train_y = train_data

    # initialize empty lists to track performance
    rmse_list = []
    r2_list = []

    # initialize tracking variables
    converged = False
    prev_mean = None
    trial = 0

    # estimate per-dimension noise once before the evaluation loop
    learned_noise = membership_privacy(train_data, run_lr, mi)

    while not converged: # evaluate until convergence
        # resample training data (bootstrap)
        sample_x, sample_y = get_samples(train_x, train_y, n_samples=int(0.5 * len(train_x)))

        rmse_sample_list = []
        r2_sample_list = []

        # OLS on fixed data is deterministic — fit once, draw noise n_draws times
        model, lr_params = run_lr((sample_x, sample_y))
        for _ in range(n_draws):
            # inject noise into a fresh copy of the weights each draw
            private_lr_params = privatize(lr_params.copy(), learned_noise)
            model.coef_ = private_lr_params[:-1]
            model.intercept_ = private_lr_params[-1]

            pred = model.predict(sample_x) # get predictions

            # add predictions to per-sample lists
            rmse_sample_list.append(root_mean_squared_error(sample_y, pred))
            r2_sample_list.append(r2_score(sample_y, pred))

        # add predictions to overall lists
        rmse_list.append(np.mean(rmse_sample_list))
        r2_list.append(np.mean(r2_sample_list))

        cur_mean = np.mean(rmse_list) # calculate current mean

        if trial % 50 == 0: # check convergence every 50 trials
            if prev_mean is not None and abs(cur_mean - prev_mean) < eta:
                converged = True
            prev_mean = cur_mean

        if trial % 10 == 0: # print every 10 trials
            print(f"Trial: {trial}, RMSE: {cur_mean:.4f}")

        trial += 1

    # compile and return overall statistics
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), np.median(rmse_list)]
    r2_stats = [np.mean(r2_list), np.std(r2_list), np.median(r2_list)]

    return rmse_stats, r2_stats, rmse_list