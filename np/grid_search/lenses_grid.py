import ssl
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.data import DataLoader

from lr_np import lrmodel, eval_sgd

import sys
sys.path.append('../')
from data_loader import gen_lenses

# ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# grid lists
# normalized
epochs = [10, 20, 30]
batch_size = [2, 4, 8, 16, 32]
learning_rate = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]

'''# non normalized
epochs = [10, 20, 30]
batch_size = [2, 4, 8, 16, 32]
learning_rate = [1e-10, 5e-10, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.05, 0.01]
'''

# get parameter combinations
combos = list(product(epochs, batch_size, learning_rate))

# load data
x, y, data_loader = gen_lenses(normalization=True)

# initialize variables to track the best parameters
best_r2 = float('-inf')
best_results = None

for (epochs, batch, lr) in combos:
    # define training objects
    train_loader = DataLoader(data_loader, batch_size=batch, shuffle=True)
    model = lrmodel(x.shape[1])
    optimizer = optim.SGD(model.parameters(), lr)
    criterion = nn.MSELoss()

    # train and evaluate 
    print(f"Training epochs {epochs}, batch size {batch}, and lr {lr}")
    r2_stats, rmse_stats = eval_sgd(model, optimizer, criterion, train_loader, x, y, epochs, eta = 1e-2)

    r2_mean = r2_stats[0]
    r2_std = r2_stats[1]
    r2_med = r2_stats[2]
    rmse_mean = rmse_stats[0]
    rmse_std = rmse_stats[1]
    rmse_med = rmse_stats[2]

    # update best results
    if r2_mean > best_r2:
        best_r2 = r2_mean
        best_results = {
            "epochs": epochs,
            "batch_size": batch,
            "learning_rate": lr,
            "r2mean": r2_mean,
            "r2std": r2_std,
            "r2med": r2_med,
            "rmsemean": rmse_mean,
            "rmsestd": rmse_std,
            "rmsemed": rmse_med
        }

    # print results
    print(f"For epochs {epochs}, batch size {batch}, and lr {lr}")
    print(f"R2 mean is {r2_mean}, R2 std is {r2_std}, and R2 median is {r2_med}")
    print("------------------------------------------------------------\n")

# print best parameters
print(f"BEST RESULTS FOR LENSES DATASET: Epochs {best_results['epochs']}, batch size {best_results['batch_size']}, and lr {best_results['learning_rate']}")
print(f"R2 mean is {best_results['r2mean']}, R2 std is {best_results['r2std']}, and R2 median is {best_results['r2med']}")
print(f"RMSE mean is {best_results['rmsemean']}, RMSE std is {best_results['rmsestd']}, and RMSE median is {best_results['rmsemed']}")