"""
==========
private.py
==========
Noise estimation algorithm for PAC membership privacy

Use PAC Privacy framework to estimate anisotropic noise for linear regression.
Optimized with SVD basis transformation to calculate optimal noise levels.
"""

import numpy as np
import torch
from numpy.linalg import svd

def get_samples(X: torch.Tensor, y: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a random subset of n_samples rows without replacement

    Args:
        X, y: feature and target
        n_samples: number of rows to sample

    Returns:
        (X[idx], y[idx]): sampled feature/target
    """
    idx = np.random.choice(X.shape[0], n_samples, replace=False)
    return X[idx], y[idx]

def compute_basis(
    data: tuple[torch.Tensor, torch.Tensor],
    mechanism: callable,
) -> np.ndarray:
    """
    Learn a projection basis (V^T) via SVD to identify the principal
    directions of variance in the mechanism's output space

    1. Run mechanism on 10000 random half-subsets of the training data
    2. Stack outputs into a matrix, center, and compute SVD
    3. Return V^T where rows are the principal directions of output variance

    Args:
        data: (train_x, train_y) tensors
        mechanism: algorithm to be privatized

    Returns:
        VT: right singular vectors capturing principal directions of variance
    """
    train_x, train_y = data
    n_samples = int(0.5 * len(train_x)) # n = half the dataset each trial
    outputs = []

    for _ in range(10000): # randomly subsample the dataset 10000 times and then average
        sampled_x, sampled_y = get_samples(train_x, train_y, n_samples)
        _, output = mechanism([sampled_x, sampled_y], *(()))
        outputs.append(output)

    outputs = np.array(outputs)

    # center outputs and compute SVD to extract principal directions
    mean_output = np.mean(outputs, axis=0)
    centered_output = outputs - mean_output
    _, _, VT = svd(centered_output, full_matrices=False)
    
    return VT # VT projection matrix for use below

def membership_privacy(
    data: tuple[torch.Tensor, torch.Tensor],
    mechanism: callable,
    mi: float,
    eta: float = 1e-3,
) -> dict[int, float]:
    """
    Estimate per-dimension noise for PAC membership privacy:
    1. Construct two neighboring datasets A and B for each training point i
    2. Run mechanism on both and measure per-dimension squared output difference
    3. Iterate until variance estimates converge per dimension (eta = 0.001)
    4. Calibrate anisotropic per-dimension noise (noise[d]) analytically

    Args:
        data: (train_x, train_y) tensors
        mechanism: black-box algorithm returning (model, weights)
        mi: Mutual Information bound used in noise calibration
        eta: per-dimension convergence threshold (default 1e-3)

    Returns:
        noise_max: dict mapping dimensions to noise across all training points
    """
    VT = compute_basis(data, mechanism) # get projection matrix VT
    train_x, train_y = data # unpack data
    noise_max = {} # store maximum noise

    # loop over every datapoint to estimate output differences:
    for i in range(len(train_x)):
        x_point, y_point = train_x[i].unsqueeze(0), train_y[i].unsqueeze(0)
        sampled_x, sampled_y = (
            torch.cat((train_x[:i], train_x[i + 1 :]), dim=0),
            torch.cat((train_y[:i], train_y[i + 1 :]), dim=0),
        ) # neighboring datasets (remove datapoint i)

        est_y = {}  # store trial result for each output dimension
        prev_vars = None  # store previous variance (for convergence check)
        trial = 0
        converged = False

        while not converged:
            x_a, y_a = get_samples(sampled_x, sampled_y, int(0.5 * len(sampled_x))) # dataset A: half sample 
            x_b, y_b = torch.cat((x_a, x_point), dim=0), torch.cat((y_a, y_point), dim=0) # dataset B: A with point i added back (neighboring)

            # get mechanism output for both datasets
            _, w_a = mechanism([x_a, y_a]) # M(A)
            _, w_b = mechanism([x_b, y_b]) # M(B)
            
            # project both outputs/weights to optimized basis using VT
            w_a, w_b = VT @ w_a, VT @ w_b

            g = (np.array(w_a) - np.array(w_b)) ** 2 # square of output difference/sensitivity, (M(A) - M(B))^2
            for idx in range(len(g)):
                est_y.setdefault(idx, []).append(g[idx]) # track per-dimension sensitivity

            if trial % 10 == 0: # convergence check
                cur_vars = np.array([np.mean(est_y[idx]) for idx in sorted(est_y)])
                if prev_vars is not None and np.all(np.abs(cur_vars - prev_vars) < eta):
                    converged = True
                prev_vars = cur_vars
            
            trial += 1
            
        # variance computations
        point_var = {idx: np.mean(est_y[idx]) for idx in est_y}  # map dimensions to empirical sensitivity in that direction
        total_sensitivity = sum(v ** 0.5 for v in point_var.values()) # sum of per-dim std., scales noise across dimensions

        # calculate noise with formula
        noise = {idx: (point_var[idx] ** 0.5 * total_sensitivity) / (4 * mi) for idx in point_var}

        # project noise back to original feature space via VT diagonal
        proj_diag = np.diag(VT.T) * np.fromiter(noise.values(), dtype=float)
        noise = {idx: proj_diag[idx] for idx in range(len(proj_diag))}

        # get max noise per dimension
        for idx, val in noise.items():
            noise_max[idx] = max(noise_max.get(idx, 0.0), val)

    print("Finished estimating noise...")
    return noise_max

def privatize(output: np.ndarray, learned_noise: dict[int, float]) -> np.ndarray:
    """
    Add learned anisotropic Gaussian noise to model weights,
    perturbing each dimension independently

    Args:
        output: flattened weight array to perturb
        learned_noise: dict mapping dimension index to noise std. dev

    Returns: output with added per-dimension Gaussian noise
    """
    scales = np.array([learned_noise[idx] for idx in range(len(output))])
    output += np.random.normal(0, scale=scales)
    
    return output