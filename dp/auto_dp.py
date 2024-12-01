import ssl
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lr_dp import lrmodel, eval_dp_lr

import sys
sys.path.append('../')
from data_loader import gen_auto

# ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

params = {
    "epochs": 30,
    "epsilon": 2.944428453,
    "delta": 1e-3,
    "norm_clip": 100000,
    "batch_size": 16
}

learning_rate = 1e-6

# load data
x, y, data_loader = gen_auto(normalization = False)

# learning objects
model = lrmodel(x.shape[1])
optimizer = optim.SGD(model.parameters(), learning_rate)
criterion = nn.MSELoss()

# train and evaluate opacus dpsgd model
print(f"Training opacus dpsgd with epsilon {params['epsilon']}:")
r2_stats, rmse_stats = eval_dp_lr(model, optimizer, criterion, data_loader, [x, y], **params)

# print information
print("For the Automobiles dataset using DPSGD with Opacus and with epsilon", params['epsilon'])
print(f"RMSE stats: {rmse_stats}")
