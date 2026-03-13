# Private LR
This repository implements **private linear regression models**, comparing **Differential Privacy (DP)** to **PAC (Probably Approximately Correct) Privacy** and a non-private baseline. [Differential privacy](https://link.springer.com/chapter/10.1007/11787006_1) is a commonly accepted privacy notion, while the [PAC Privacy framework](https://eprint.iacr.org/2024/718) is more recent.

We use Python to implement:
* **DPSGD-LR** — DP stochastic gradient descent using [Opacus](https://opacus.ai/)
* **PAC-LR** — PAC Privacy framework with custom anisotropic noise estimation
* **OLS baseline** — Non-private scikit-learn linear regression

Models are evaluated on the **membership inference task**, where an attacker attempts to infer whether a given point was included in training. Utility is quantified via R² and RMSE, averaged across multiple trials to capture both stability and overall performance.

Privacy is parameterized by **posterior success rate (PSR)** — the adversary's probability of correct membership inference. Since ε can be analytically translated into PSR, the two methods can be compared under identical conditions.

## Privacy Mechanisms

### PAC-LR
Unlike DPSGD-LR's isotropic (uniform) noise, PAC-LR is data-aware: it uses SVD to identify per-dimension sensitivity and calibrates noise accordingly. The algorithm:
1. Empirically estimate per-dimension sensitivity using projection matrix V^T from SVD
2. Iteratively compute noise based on the Mutual Information (MI) privacy parameter until convergence threshold η is reached
3. Project the calibrated noise back to the original feature space
4. Apply Gaussian perturbation

### DPSGD-LR
Privacy via training loop — each epoch, for each batch:
1. Compute per-sample gradients
2. Clip gradients to threshold C (bounding influence of any single data point)
3. Aggregate and perturb with Gaussian noise scaled to C and ε
4. Update parameters and track privacy budget (via Opacus)

## Repository Structure

```
├── data_loader.py          # Dataset loading and preprocessing
├── datasets/
│   ├── lenses.csv          # 24 instances, 4 features — contact lens class
│   ├── concrete.csv        # 103 instances, 7 features — compressive strength (MPa)
│   └── auto.csv            # 201 instances, 15 features — car price (USD)
├── np/
│   ├── lr_np.py            # OLS evaluation (bootstrap)
│   └── run_np.py           # Runner: non-private baseline
├── dp/
│   ├── lr_dp.py            # DPSGD-LR training loop
│   ├── private.py          # Opacus privatization wrapper
│   ├── run_dp.py           # Runner: DPSGD-LR
│   └── grid_search.py      # Hyperparameter search for DPSGD-LR
└── pac/
    ├── lr_pac.py           # PAC-LR training loop
    ├── private.py          # Anisotropic noise estimation (SVD + MI)
    ├── run_pac.py          # Runner: PAC-LR
    └── alpha_search.py     # Hyperparameter search for Ridge/Lasso regularization (α)
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/hillaryyang/priv_lr.git
cd priv_lr
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each model has a runner script with an optional `--dataset` flag. Omitting it runs all three datasets.

**Non-private OLS baseline**
```bash
cd np
python run_np.py                        # run all datasets
python run_np.py --dataset lenses       # run only lenses
```

**DPSGD-LR**
```bash
cd dp
python run_dp.py                        # run all datasets
python run_dp.py --dataset concrete     # run only concrete
```

**PAC-LR**
```bash
cd pac
python run_pac.py                       # run all datasets
python run_pac.py --dataset auto        # run only automobiles
```

Results (RMSE and R² mean, std. dev, median) are printed to the command line.

## Hyperparameter Search

**DPSGD-LR** — grid search over epochs, clipping norm, and batch size:
```bash
cd dp
python grid_search.py --dataset lenses
```

**PAC-LR** — grid search over Ridge/Lasso regularization parameter α:
```bash
cd pac
python alpha_search.py --dataset lenses
```

## Datasets

All datasets are sourced from the [UCI ML Repository](https://archive.ics.uci.edu/) and normalized with `StandardScaler` before training.

| Dataset | Instances | Features | Target |
|---------|-----------|----------|--------|
| [Lenses](https://archive.ics.uci.edu/dataset/58/lenses) | 24 | 4 | contact lens class |
| [Concrete](https://archive.ics.uci.edu/dataset/182/concrete+slump+test) | 103 | 7 | compressive strength (MPa) |
| [Automobiles](https://archive.ics.uci.edu/dataset/10/automobile) | 201 | 15 | price (USD) |
