from lr_np import eval_ols

import sys
sys.path.append('../')
from data_loader import gen_lenses

# get data
x, y, data_loader = gen_lenses(norm = True)

# get private r2 values averaged over 10k trainings
rmse_stats, r2_stats = eval_ols(x, y)

print(f"For the non-private Lenses dataset")
print(f"RMSE stats {rmse_stats}, R2 stats: {r2_stats}")