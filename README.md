# Private Linear Regression

This repository implements and compares three linear regression models under a **membership inference threat model**: a non-private OLS baseline, **DPSGD-LR** (Differential Privacy via [Opacus](https://opacus.ai/)), and **PAC-LR** (PAC Privacy with custom anisotropic noise estimation).

[Differential Privacy](https://link.springer.com/chapter/10.1007/11787006_1) is a well-established privacy framework; the [PAC Privacy framework](https://eprint.iacr.org/2024/718) is more recent and data-aware.

## Threat Model

The adversary performs a **membership inference attack**: given a trained model and a data point, they attempt to determine whether that point was included in the training set. The adversary's advantage is measured by **Posterior Success Rate (PSR)**, which is the probability of a correct membership decision. PSR = 0.5 is random guessing (full privacy); PSR = 1.0 means the adversary always succeeds (no privacy).

## Privacy Parameterization

Both mechanisms are evaluated at the same PSR to enable direct comparison. We chose seven distinct, evenly spaced PSRs (0.52, 0.55, 0.65, 0.75, 0.85, 0.95, 0.98) in order to test a variety of privacy levels. 

**DPSGD-LR** uses (ε, δ)-DP Differential Privacy, where epsilon (ε) is the privacy parameter and delta (δ) is the probability of a privacy breach. ε which can be derived analytically from PSR [Xiao 2023](https://arxiv.org/abs/2210.03458):

```
ε = ln((1 - δ) / (1 - PSR) - 1)
```

## Privacy Mechanisms

### DPSGD-LR

Privacy is enforced during the SGD training loop via Opacus. Each epoch, for each batch:

1. Compute per-sample gradients
2. Clip each gradient to norm threshold C (bounding any single point's influence)
3. Aggregate and add Gaussian noise scaled to C and ε
4. Update parameters; Opacus tracks the cumulative privacy budget

### PAC-LR

Privacy is enforced by perturbing the final OLS weights with **anisotropic** (data-aware) Gaussian noise:

1. Empirically estimate per-dimension sensitivity using projection matrix V^T from singular value decomposition (VSD)
2. Iteratively compute noise based on the Mutual Information (MI) privacy parameter until convergence threshold η is reached
3. Project the calibrated noise back to the original feature space
4. Apply Gaussian perturbation

Because noise is scaled to each dimension's empirical sensitivity, PAC-LR achieves a tighter privacy-utility trade-off than isotropic mechanisms (DP).

## Evaluation Methodology

All three models are evaluated via **bootstrap sampling** (50% of training data, without replacement) to simulate realistic variability. Each runner (`run_np.py`, `run_dp.py`, `run_pac.py`) loops until the cumulative RMSE mean stabilizes within convergence threshold η = 0.01 (checked every 50 trials). Final results report RMSE/R^2 evaluation statistics, including the mean, standard deviation, and median.

## Repository Structure

```
├── data_loader.py          # Dataset loading, preprocessing, and normalization
├── datasets/
│   ├── lenses.csv          # Contact lenses dataset
│   ├── concrete.csv        # Concrete dataset
│   └── auto.csv            # Automobiles dataset
├── np/
│   ├── lr_np.py            # OLS bootstrap evaluation
│   └── run_np.py           # Runner: non-private baseline
├── dp/
│   ├── lr_dp.py            # DPSGD-LR training and convergence loop
│   ├── private.py          # Opacus privatization wrapper
│   ├── run_dp.py           # Runner: DPSGD-LR
│   └── grid_search.py      # Hyperparameter search (epochs, clipping norm, batch size, lr)
└── pac/
    ├── lr_pac.py           # PAC-LR training and convergence loop
    ├── private.py          # SVD basis estimation + anisotropic noise calibration
    └── run_pac.py          # Runner: PAC-LR
```

## Installation

1. Download and `cd` into the repository

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

Each runner accepts an optional `-d` flag to select a dataset to run by name.

**Non-private OLS baseline**
```bash
cd np
python run_np.py                  # run all datasets
python run_np.py -d [concrete | lenses | auto]
```

**DPSGD-LR**
```bash
cd dp
python run_dp.py                  # run all datasets
python run_dp.py -d [concrete | lenses | auto] 
```

**PAC-LR**
```bash
cd pac
python run_pac.py                 # run all datasets
python run_pac.py -d [concrete | lenses | auto]
```

Results (RMSE and R^2 mean, std. dev, median) are printed to the command line.

## Hyperparameter Search

**DPSGD-LR** — grid search over epochs, clipping norm, batch size, and learning rate:
```bash
cd dp
python grid_search.py -d [concrete | lenses | auto]
```

## Datasets

All datasets are sourced from the [UCI ML Repository](https://archive.ics.uci.edu/) and normalized to zero mean and unit variance with `StandardScaler` before training.

| Dataset | Instances | Features | Target |
|---------|-----------|----------|--------|
| [Lenses](https://archive.ics.uci.edu/dataset/58/lenses) | 24 | 4 | contact lens class |
| [Concrete](https://archive.ics.uci.edu/dataset/182/concrete+slump+test) | 103 | 7 | compressive strength (MPa) |
| [Automobiles](https://archive.ics.uci.edu/dataset/10/automobile) | 201 | 15 | price (USD) |
