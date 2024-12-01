# for calculating range of gradients for grid search
import numpy as np
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../')
from data_loader import gen_lenses

class lrmodel(nn.Module):
 def __init__(self, input_dim):
   super(lrmodel, self).__init__()
   self.linear = nn.Linear(input_dim, 1)

 def forward(self, x):
   return self.linear(x)
 
x, y, data_loader = gen_lenses(normalization=False)

model = lrmodel(x.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.000001)

gradient_norms = []

# get gradient norms during training
for inputs, labels in data_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item()
            gradient_norms.append(param_norm.item())

gradient_norms = np.array(gradient_norms)

median_norm = np.median(gradient_norms)
mean_norm = np.mean(gradient_norms)

# print the results
print(f"median norm: {median_norm:.4f}")
print(f"mean norm: {mean_norm:.4f}")
