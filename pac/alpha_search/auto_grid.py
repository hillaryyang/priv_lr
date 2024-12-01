import math
from lr_pac import membership_pac

import sys
sys.path.append('../../')
from data_loader import gen_auto

# get data
x, y, _ = gen_auto(norm=True)

# calculate mutual information requirement
post_success = 0.55
mi = post_success * math.log(2 * post_success) + (1 - post_success) * math.log(2 - 2 * post_success)

alphas = []
for i in range(-10, 10):
    alphas.append(2 ** i)

best_rmse = float('inf')
best_params = None

print(f"Grid search for post. success rate {post_success}")

for alpha_val in alphas:
    print(f"Training alpha {alpha_val}...")
    _, rmse_stats = membership_pac([x, y], mi, False, alpha_val)

    rmse_mean = rmse_stats[0]
    rmse_std = rmse_stats[1]
    rmse_med = rmse_stats[2]

    if rmse_mean < best_rmse:
        best_rmse = rmse_mean
        best_params = {
            "best_alpha": alpha_val,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_med": rmse_med
        }
    
print(f"For the Automobile dataset, with PSR {post_success}, the best alpha value is {best_params['best_alpha']}")
print(f"RMSE mean: {best_params['rmse_mean']}, RMSE mean: {best_params['rmse_std']}, RMSE median: {best_params['rmse_med']}\n")