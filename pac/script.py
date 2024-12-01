# post success rates
psr = [0.52, 0.55, 0.65, 0.75, 0.85, 0.95, 0.98]

# LASSO
# lenses
# alphas = [0.5, 0.5, 0.0625, 0.00390625, 0.001953125, 0.00390625, 0.015625]
# concrete
# alphas = [0.125, 0.03125, 0.00390625, 0.001953125, 0.0009765625, 0.0009765625, 0.0009765625]
# auto
# alphas = [0.125, 0.03125, 0.00390625, 0.001953125, 0.0009765625, 0.0009765625, 0.0009765625]

# RIDGE
# lenses
# alphas = [512, 32, 2, 0.25, 0.03125, 0.015625, 0.0625]
# concrete
# alphas = [8, 8, 0.25, 0.5, 0.125, 0.0009765625, 0.03125]
# auto
alphas = [128, 32, 4, 0.5, 0.03125, 0.03125, 0.0625]

# dataset
dataset = "auto"

if dataset == "auto": 
    ch = "p"
else:
    ch = dataset[0]

# directories
base_dir = "/N/u/haiyang/Quartz/slate/privacy1/final/ridge/{}".format(dataset)
script_dir = "/N/slate/haiyang/privacy1/final/ridge/{}".format(dataset)

# python script
python_template = '''import math
import numpy as np
from lr_pac import membership_pac

import sys
sys.path.append('../../')
from data_loader import gen_{dataset}

# get data
x, y, _ = gen_{dataset}(normalization=True)

# calculate mutual information requirement using equation 3 in paper
post_success = {psr}
alpha_val = {alpha}
mi = post_success * math.log(2 * post_success) + (1 - post_success) * math.log(2 - 2 * post_success)

final_vals = []

# number of runs to average over
for i in range(50):
    print(f"Running trial {{i}}...")
    _, rmse_stats = membership_pac([x, y], mi, True, alpha_val)

    # append mean to running list
    rmse_avg = rmse_stats[0]
    final_vals.append(rmse_avg)
    print(f"For run {{i}}, the RMSE average was {{rmse_avg}}")

final_avg = np.mean(final_vals)
final_std = np.std(final_vals)

# print everything
print(f"For the {dataset} dataset, with PSR {{post_success}}")
print(f"RMSE avg: {{final_avg}}, RMSE std: {{final_std}}")
print(f"Data: {{final_vals}}")'''

# sh scripts
cob_template = '''#!/bin/bash
cd {base_dir}
python -m venv venv
source venv/bin/activate
module load python/gpu
export PYTHONUNBUFFERED=TRUE
python {ch}{index}.py
'''

mycob_template = '''#!/bin/bash
#SBATCH -J {ch}{index}
#SBATCH -p gpu
#SBATCH -A staff
#SBATCH -o {ch}{index}%j.txt
#SBATCH -e {ch}{index}%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hillaryyang2@gmail.com
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
export PYTHONUNBUFFERED=TRUE
# Load any modules that your program needs
cd {script_dir}
srun {ch}ob{index}.sh
'''

# generate the files
for idx, psr in enumerate(psr, start=1):
    python_filename = f'{ch}{idx}.py'
    with open(python_filename, 'w') as f:
        f.write(python_template.format(psr=psr, dataset=dataset, alpha=alphas[idx-1]))
    
    cob_filename = f'{ch}ob{idx}.sh'
    with open(cob_filename, 'w') as f:
        f.write(cob_template.format(base_dir=base_dir, index=idx, ch = ch, dataset=dataset))
    
    mycob_filename = f'my{ch}ob{idx}.sh'
    with open(mycob_filename, 'w') as f:
        f.write(mycob_template.format(index=idx, script_dir=script_dir, ch = ch))

print("Files generated successfully.")
