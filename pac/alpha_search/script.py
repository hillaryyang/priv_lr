# post success rates
psr = [0.52, 0.55, 0.65, 0.75, 0.85, 0.95, 0.98]

# dataset
dataset = "auto"

if dataset == "auto": 
    ch = "p"
else:
    ch = dataset[0]

# directories
base_dir = "/N/u/haiyang/Quartz/slate/privacy1/final/{}".format(dataset)
script_dir = "/N/slate/haiyang/privacy1/final/{}".format(dataset)

# python script
python_template = '''import math
from lr_pac import membership_pac

import sys
sys.path.append('../')
from data_loader import gen_{dataset}

# get data
x, y, _ = gen_{dataset}(normalization=True)

# calculate mutual information requirement using equation 3 in paper
post_success = {psr}
mi = post_success * math.log(2 * post_success) + (1 - post_success) * math.log(2 - 2 * post_success)

alphas = []
for i in range(-10, 10):
    alphas.append(2 ** i)

best_rmse = float('inf')
best_params = None

print(f"Grid search for post. success rate {{post_success}}")

# get private r2 values averaged over 10k trainings
for alpha_val in alphas:
    print(f"Training alpha {{alpha_val}}...")
    _, rmse_stats = membership_pac([x, y], mi, True, alpha_val)

    # unpack stats
    rmse_mean = rmse_stats[0]
    rmse_std = rmse_stats[1]
    rmse_med = rmse_stats[2]

    # check if this is the best value so far
    if rmse_mean < best_rmse:
        best_rmse = rmse_mean
        best_params = {{
            "best_alpha": alpha_val,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_med": rmse_med
        }}
    
# print everything
print(f"For the {dataset} dataset, with PSR {{post_success}}, the best alpha value is {{best_params['best_alpha']}}")
print(f"RMSE mean: {{best_params['rmse_mean']}}, RMSE stdev: {{best_params['rmse_std']}}, RMSE median: {{best_params['rmse_med']}}\\n")'''

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
        f.write(python_template.format(psr=psr, dataset=dataset))
    
    cob_filename = f'{ch}ob{idx}.sh'
    with open(cob_filename, 'w') as f:
        f.write(cob_template.format(base_dir=base_dir, index=idx, ch = ch, dataset=dataset))
    
    mycob_filename = f'my{ch}ob{idx}.sh'
    with open(mycob_filename, 'w') as f:
        f.write(mycob_template.format(index=idx, script_dir=script_dir, ch = ch))

print("Files generated successfully.")
