from lr_np import eval_ols

import sys
sys.path.append('../')
from data_loader import gen_auto

# get data
x, y, data_loader = gen_auto(normalization = True)

# evaluate the ols model, sample half
rmse_stats, r2_stats = eval_ols(x, y)

# print r2
print("For the Automobile dataset, non-private:")
print(f"RMSE stats {rmse_stats}, R2 stats: {r2_stats}")