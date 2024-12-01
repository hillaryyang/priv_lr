import numpy as np
import statistics
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score, root_mean_squared_error
from private import membership_privacy, privatize, get_samples_safe


def run_lr(data, alpha=None, config=None, seed=None):
    """
    data: [train_x, train_y]
    config: None
    seed: random seed
    Run linear regression on the training data
    Return: 
        the learned model
        the learned weights in numpy array \\
        (num_features+1,) the last element is the intercept
    """
    train_x, train_y = data

    # train the model
    # model = LinearRegression()
    model = Ridge(alpha)
    # model = Lasso(alpha)
    model.fit(train_x, train_y)

    weight = model.coef_  # [num_features, ]
    intercept = model.intercept_  # [1, ]

    if weight.ndim == 2 and intercept.ndim == 1:
        intercept = intercept.reshape(1, -1)

    tot_weights = np.hstack((weight, intercept)).flatten()
    return model, tot_weights


def train_lr(data, config=None, seed=None):
    """
    data: [train_x, train_y, num_classes]
    config: None
    seed: random seed
    Run linear regression on the training data
    Return: the learned model
    """
    # train the model
    model, lr_params = run_lr(data, config, seed)

    return model

def membership_pac(train_data, mi, learn_basis, alpha=None, eta = 1e-2):
    """
    train_data: [train_x, train_y, num_classes]
    config: configuration for LR (regularization)
    n_trials: number of trials to evaluate the model
    Run linear regression on the training train_data with PAC
    Return: RMSE statistics
    """
    train_x, train_y = train_data

    # lists to store avg stats for each sampling
    r2_list = []
    rmse_list = []

    # convergence parameters
    converged = False
    prev_mean = float("inf")
    trial = 0

    # add noise
    if alpha == None:
        # print(f"Learning PAC membership noise with OLS...")
        learned_noise = membership_privacy(train_data, run_lr, mi, learn_basis)
    else:
        # print(f"Learning PAC membership noise with regularization...")
        learned_noise = membership_privacy(train_data, run_lr, mi, learn_basis, alpha)

    while not converged:
        # sample the training data
        _train_x, _train_y = get_samples_safe(train_x, train_y, n_samples=len(train_x))

        r2_sampling = []
        rmse_sampling = []

        # train the model and average for each sampling
        for i in range(1000):
            model, lr_params = run_lr([_train_x, _train_y], alpha)
            private_lr_params = privatize(lr_params, learned_noise)
                        
            # set the weights with private parameters
            model.coef_ = private_lr_params[:-1]
            model.intercept_ = private_lr_params[-1]

            # evaluate the model
            pred = model.predict(_train_x)

            # get the rmse and r2 values
            rmse_val = root_mean_squared_error(_train_y, pred)
            r2_val = r2_score(_train_y, pred)

            # add to lists
            rmse_sampling.append(rmse_val)
            r2_sampling.append(r2_val)

        # add values for this sampling to list
        r2_list.append(np.mean(r2_sampling))
        rmse_list.append(np.mean(rmse_sampling))

        # print(f"trial: {trial}, last avg added: {np.mean(rmse_sampling)}")

        if trial == 20 and cur_mean > 2:
            converged = True

        # convergence testing
        cur_mean = np.mean(rmse_list)
        if trial % 50 == 0:
            if abs(cur_mean - prev_mean) < eta:
                converged = True
            else:
                prev_mean = cur_mean

        if trial % 10 == 0:
           print(f"Trial: {trial}, RMSE mean: {np.mean(rmse_list)}")

        trial += 1

    # stats
    r2_stats = [np.mean(r2_list), np.std(r2_list), statistics.median(r2_list)]
    rmse_stats = [np.mean(rmse_list), np.std(rmse_list), statistics.median(rmse_list)]

    # return the stats
    return r2_stats, rmse_stats