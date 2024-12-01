# epsilons
epsilons = [0.08002347672, 0.2006525135, 0.6190238237, 1.098598955, 1.734589291, 2.944428453, 3.891810094]

# dataset
dataset = "auto"

if dataset == "auto": 
    ch = "p"
else:
    ch = dataset[0]

# directories
base_dir = "/N/u/haiyang/Quartz/slate/privacy1/final/dpsgd/{}".format(dataset)
script_dir = "/N/slate/haiyang/privacy1/final/dpsgd/{}".format(dataset)

# python script
python_template = '''import ssl
import numpy as np
import warnings
warnings.simplefilter("ignore")
import torch.nn as nn
import torch.optim as optim

from lr_dp import lrmodel, eval_dp_lr

import sys
sys.path.append('../../')
from data_loader import gen_{dataset}

# ignore ssl certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# optimal parameters from grid search
params = {{
    "epochs": {epochs},
    "epsilon": {eps},
    "delta": 1e-3,
    "norm_clip": {nc},
    "batch_size": {bs}
}}

learning_rate = {lr}

# load data
x, y, data_loader = gen_{dataset}(normalization = True)

# train and evaluate opacus DP SGD model
print(f"Training opacus DP SGD with epsilon {eps}:")

final_vals = []

for i in range(20):
    print(f"Starting run {{i}}...")
    # learning objects
    model = lrmodel(x.shape[1])
    optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # evaluate
    _, rmse_stats = eval_dp_lr(model, optimizer, criterion, data_loader, [x, y], **params)
    
    # append mean to running list
    rmse_avg = rmse_stats[0]
    final_vals.append(rmse_avg)
    print(f"For run {{i}}, the RMSE average is {{rmse_avg}}")

final_avg = np.mean(final_vals)
final_std = np.std(final_vals)

# print information
print("For the {dataset} dataset using DPSGD with Opacus and with epsilon", {eps})
print(f"RMSE mean: {{final_avg}}, RMSE std: {{final_std}}")
print(f"Data points are {{final_vals}}")'''

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
#SBATCH -J {ch}dp{index}
#SBATCH -p gpu
#SBATCH -A staff
#SBATCH -o {ch}dp{index}%j.txt
#SBATCH -e {ch}dp{index}%j.err
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
for idx, eps in enumerate(epsilons, start=1):
    python_filename = f'{ch}{idx}.py'
    with open(python_filename, 'w') as f:
        f.write(python_template.format(eps=eps, dataset=dataset, epochs=10, nc = 1, bs = 16, lr = 0.01))

    cob_filename = f'{ch}ob{idx}.sh'
    with open(cob_filename, 'w') as f:
        f.write(cob_template.format(base_dir=base_dir, index=idx, ch = ch, dataset=dataset))
    
    mycob_filename = f'my{ch}ob{idx}.sh'
    with open(mycob_filename, 'w') as f:
        f.write(mycob_template.format(index=idx, script_dir=script_dir, ch = ch))

print("Files generated successfully.")
