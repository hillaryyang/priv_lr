import ssl
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.data import DataLoader

from lr_dp import lrmodel, eval_dp_lr

import sys
sys.path.append('../../')
from data_loader import gen_concrete

# ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# parameter grid
'''# normalized
epochs = [10, 20, 30]
norm_clip = [0.5, 1, 2, 4]
batch_size = [2, 4, 8, 16]
learning_rate = [0.005, 0.01, 0.05, 0.1]'''

'''# non normalized
epochs = [10, 20, 30]
norm_clip = [1000, 5000, 10000, 50000, 100000, 500000]
batch_size = [2, 4, 8, 16]
learning_rate = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6]'''

#non normalized
epochs = [1, 5, 10, 20, 30]
norm_clip = []
batch_size = [2, 4, 8, 16, 32]
learning_rate = []

for i in range(0, 7):
    norm_clip.append(10 ** i)

for i in range(-8, -1):
    learning_rate.append(10 ** i)


# get parameter combinations
combos = list(product(epochs, norm_clip, batch_size, learning_rate))

# epsilon_list = [0.08002347672, 0.2006525135, 0.6190238237, 1.098598955, 1.734589291, 2.944428453, 3.891810094]
epsilon = 2.944428453
delta = 1e-3

# load data
x, y, data_loader = gen_concrete(normalization=False)

print(f"For epsilon {epsilon}:")

# initialize variables to track the best parameters
best_rmse = float('inf')
best_params = None

for (epoch, nc, batch, lr) in combos:
    # training objects
    train_loader = DataLoader(data_loader, batch_size=batch, shuffle=True)
    model = lrmodel(x.shape[1])
    optimizer = optim.SGD(model.parameters(), lr)
    criterion = nn.MSELoss()

    # set up parameters for Opacus
    params = {
        "epochs": epoch,
        "epsilon": epsilon,
        "delta": delta,
        "norm_clip": nc,
        "batch_size": batch
    }

    # train and evaluate the model
    print(f"Training epochs={epoch}, norm_clip={nc}, batch_size={batch}, learning_rate={lr}")
    _, rmse_stats = eval_dp_lr(model, optimizer, criterion, data_loader, [x, y], **params)
    
    # unpack stats
    rmse_mean = rmse_stats[0]
    rmse_std = rmse_stats[1]
    rmse_med = rmse_stats[2]

    # check if this is the best model so far
    if rmse_mean < best_rmse:
        best_rmse = rmse_mean
        best_params = {
            "epochs": epoch,
            "norm_clip": nc,
            "batch_size": batch,
            "learning_rate": lr,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_med": rmse_med
        }

    # write results for this combination
    print(f"Results for epochs={epoch}, norm clip={nc}, batch size={batch}, learning rate={lr}:\n")
    print(f"RMSE mean: {rmse_mean }, R2 std: {rmse_std}, R2 median: {rmse_med}\n")
    print("------------------------------------------------------------\n")

# write the best combination of parameters for this epsilon
print(f"Best parameters found for epsilon: {epsilon} on the Concrete dataset\n")
print(f"Epochs: {best_params['epochs']}, Norm Clip: {best_params['norm_clip']}, Batch Size: {best_params['batch_size']}, Learning Rate: {best_params['learning_rate']}\n")
print(f"RMSE mean: {best_params['rmse_mean']}, RMSE mean: {best_params['rmse_std']}, RMSE median: {best_params['rmse_med']}\n")