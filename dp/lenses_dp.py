import ssl
import numpy as np
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.optim as optim

from lr_dp import lrmodel, eval_dp_lr

import sys
sys.path.append('../')
from data_loader import gen_lenses

# ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# optimal parameters from grid search
params = {
    "epochs": 10,
    "epsilon": 1.098598955,
    "delta": 1e-3,
    "norm_clip": 10,
    "batch_size": 16
}

learning_rate = 1e-5

# load data
x, y, data_loader = gen_lenses(normalization = False)

# train and evaluate opacus dpsgd model
print(f"Training opacus dpsgd with epsilon {params['epsilon']}:")

final_vals = []

# number of runs to average over
for i in range(1):
    # learning objects
    model = lrmodel(x.shape[1])
    optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # evaluate
    _, rmse_stats = eval_dp_lr(model, optimizer, criterion, data_loader, [x, y], **params)
    
    # append mean to running list
    final_vals.append(rmse_stats[0])

final_avg = np.mean(final_vals)
final_std = np.std(final_vals)

# print information
print("For the Lenses dataset using DPSGD with Opacus and with epsilon", params['epsilon'])
print(f"RMSE mean: {final_avg}, RMSE std: {final_std}")
