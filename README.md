# Private LR
This repository implements **private linear regression models**, comparing **Differential Privacy (DP)** to **PAC (Probably Approximately Correct) Privacy** and a non-private baseline. [Differential privacy](https://link.springer.com/chapter/10.1007/11787006_1) is a commonly accepted privacy notion, while [PAC Privacy framework](https://eprint.iacr.org/2024/718) is more recent.

## Source Code
The source code is located in GitHub, at the link: [https://github.com/hillaryyang/priv_lr](https://github.com/hillaryyang/priv_lr)

## Features
We use Python to implement:
* DP stochastic gradient descent (DPSGD-LR) using [Opacus](https://opacus.ai/)
* PAC Privacy framework (PAC-LR)

as well as non-private baselines. 

## Repository Structure
* `datasets/`: Contains the 3 datasets (Lenses, Concrete, Automobiles) used for training and testing
    * [Lenses](https://archive.ics.uci.edu/dataset/58/lenses): 24 instances and 3 features; predict class
    * [Concrete](https://archive.ics.uci.edu/dataset/182/concrete+slump+test): 103 instances and 7 features; predict concrete compressive strength
    * [Automobiles](https://archive.ics.uci.edu/dataset/10/automobile): 201 instances and 15 features; predict car price
* `np/`: Code for non private baseline (OLS with scikit-learn)
* `dp/`: Code for DPSGD-LR  
    * `dp/private.py`: Updates training objects using Opacus' privacy engine
    * `dp/grid_search/`: Contains scripts for grid search for optimal hyperparameters
* `pac/`: Code for PAC-LR
    * `pac/private.py`: Privatization functions such as noise estimation for membership privacy
    * `pac/alpha_search/`: Scripts for tuning hyperparameter alpha for regularization (Ridge and Lasso)
* `data_loader.py`: Utility script for loading/preprocessing datasets

## Installation
1. Clone the repository:
```
git clone https://github.com/hillaryyang/priv_lr.git
cd priv_lr
```

2. Create and activate a virtual environment
```
python -m venv ~/env
source ~/env/bin/activate  
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Usage
After inputting the desired hyperparameters in the Python files, run the code for DP/PAC/non-private:
* DP: `python3 dp/<dataset>_dp.py`
* PAC: `python3 pac/<dataset>_pac.py`
* Non-private: `python3 np/<dataset>_np.py`

Results are printed to the command line.
