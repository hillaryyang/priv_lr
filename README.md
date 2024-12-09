# Private LR
This repository implements **private linear regression models**, comparing **Differential Privacy (DP)** to **PAC (Probably Approximately Correct) Privacy** and a non-private baseline. [Differential privacy](https://link.springer.com/chapter/10.1007/11787006_1) is a commonly accepted privacy notion, and the [PAC Privacy framework](https://eprint.iacr.org/2024/718) is more recent.

## Source Code
The source code is located in GitHub, at the link: [https://github.com/hillaryyang/priv_lr](https://github.com/hillaryyang/priv_lr)

## Features
We use Python to implement
* DP stochastic gradient descent using [Opacus](https://opacus.ai/)
* PAC Privacy framework

as well as non-private baselines. 

## Repository Structure
* `datasets/`: Contains the 3 datasets (Lenses, Concrete, Automobiles) used for training and testing
* `np/`: Code for non private baseline
* `dp/`: Code for DP linear regression  
    * `dp/private.py`: Updates training objects using Opacus' privacy engine
    * `dp/grid_search/`: Contains scripts for grid search for DP SGD optimal hyperparameters
* `pac/`: Code for PAC linear regression
    * `pac/private.py`: Privatization functions such as noise estimation for membership privacy
    * `pac/alpha_search/`: Scripts for tuning hyperparameter alpha for regularized linear regression
* `data_loader.py`: Utility script for loading/preprocessing datasets

## Installation
1. Clone the repository:
```
git clone https://github.com/hillaryyang/priv_lr.git
cd priv_lr
```

2. For MacOS, create and activate a virtual environment
```
python -m venv ~/env
source ~/venv/bin/activate  
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
