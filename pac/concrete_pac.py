import math
from lr_pac import membership_pac

import sys
sys.path.append('../')
from data_loader import gen_concrete

# get data
x, y, _ = gen_concrete(normalization=True)

# calculate mutual information requirement using equation 3 in paper
post_success = 0.95
mi = post_success * math.log(2 * post_success) + (1 - post_success) * math.log(2 - 2 * post_success)

_, rmse_stats = membership_pac([x, y], mi, True, alpha=512)

# print everything
print(f"For the Concrete dataset, with PSR {post_success}")
print(f"RMSE stats: {rmse_stats}")