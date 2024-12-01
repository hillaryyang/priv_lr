import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score


from lr_np import lrmodel, eval_sgd

import sys
sys.path.append('../')
from data_loader import gen_orders

# set parameters
epochs = 30
batch_size = 2
learning_rate = 0.15

# generate data
x, y, data_loader = gen_orders(normalization=True)

# define model, optimizer, criterion, and train_loader for sgd
model = lrmodel(x.shape[1])
optimizer = optim.SGD(model.parameters(), learning_rate)
criterion = nn.MSELoss()
train_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)

print(f"For epochs {epochs}, batch size {batch_size}, and lr {learning_rate}")

# evaluate the sgd model
r2_mean, r2_std, r2_med = eval_sgd(model, optimizer, criterion, train_loader, x, y, epochs, eta = 1e-2)

'''sgd = SGDRegressor(max_iter=1000, tol=1e-6, eta0=5e-7)
sgd.fit(x, y)
sgd_predict = sgd.predict(x)

r2_mean = r2_score(y, sgd_predict)'''

# print mse and r2 for OLS and SGD
print("For non-private SGD:")
# print(r2_mean)
print(f"R2 mean is {r2_mean}, R2 std is {r2_std}, and R2 median is {r2_med}")