import numpy as np
import torch
from util import get_samples_safe
from numpy.linalg import svd

def compute_basis(data, mechanism, alpha=None, config=None):
    train_x, train_y = data
    n_samples = int(0.5 * len(train_x))
    outputs = []

    for i in range(10000):
        # sample dataset
        sampled_x, sampled_y = get_samples_safe(train_x, train_y, n_samples)

        # get the output of the black-box mechanism
        _, output = mechanism([sampled_x, sampled_y], alpha)

        outputs.append(output)
    
    outputs = np.array(outputs)
    mean_output = np.mean(outputs, axis=0)
    centered_output = outputs - mean_output
    U, Sigma, VT  = svd(centered_output, full_matrices=False) 
    return VT

def membership_privacy(data, mechanism, mi, learn_basis, alpha=None, eta = 1e-3):
    if learn_basis == True:
        VT = compute_basis(data, mechanism, alpha)
    noise_max = {} # store maximum noise

    train_x, train_y = data

    for i in range(len(train_x)):
        # store ith data point to add back in later
        x_point = train_x[i].unsqueeze(0)
        y_point = train_y[i].unsqueeze(0)

        # remove ith data point
        sampled_x = torch.cat((train_x[:i], train_x[i+1:]), dim=0)
        sampled_y = torch.cat((train_y[:i], train_y[i+1:]), dim=0)

        est_y = {}  # store the trial result for each output dimension
        prev_vars = None  # store the previous variance for each output dimension
        trial = 0 # count trials
        converged = False

        while not converged:
            # sample half of data
            a_x, a_y = get_samples_safe(sampled_x, sampled_y, int(0.5 * len(sampled_x)))

            # get neighboring dataset to A (B)
            b_x = torch.cat((a_x, x_point), dim = 0)
            b_y = torch.cat((a_y, y_point), dim = 0)

            # get output from mechanism for both datasets
            if alpha == None:
                _, w_a = mechanism([a_x, a_y])
                _, w_b = mechanism([b_x, b_y])
            else:
                _, w_a = mechanism([a_x, a_y], alpha)
                _, w_b = mechanism([b_x, b_y], alpha)   

            if learn_basis:
                w_a = VT @ w_a
                w_b = VT @ w_b

            # calculate difference squared for each dimension between A and B
            g = [(a - b) ** 2 for a, b in zip(w_a, w_b)]

            # store difference for each dimension
            for idx in range(len(g)):
                if idx not in est_y:
                    est_y[idx] = []
                est_y[idx].append(g[idx])

            # check for convergence
            if trial % 10 == 0:
                cur_vars = {idx: np.mean(est_y[idx]) for idx in est_y}
                if prev_vars is None:
                    prev_vars = {}
                    for idx in est_y:
                        prev_vars[idx] = cur_vars[idx]
                else:
                    converged = True
                    for idx in est_y:
                        if abs(cur_vars[idx] - prev_vars[idx]) > eta:
                            converged = False
                            if trial % 1000 == 0:
                                print(f"n_iter: {trial}, idx: {idx}, var diff: {abs(cur_vars[idx] - prev_vars[idx])}")
                    if not converged:
                        prev_vars = {idx: cur_vars[idx] for idx in cur_vars}
            trial += 1
            
        # compute noise for this data point
        final_var = {idx: np.mean(est_y[idx]) for idx in est_y}

        sqrt_total_var = sum([final_var[idx] ** 0.5 for idx in final_var])
        noise = {}

        for idx in final_var:
            denominator = 4 * mi
            numerator = final_var[idx] ** 0.5 * sqrt_total_var
            noise[idx] = numerator / denominator
        
        if learn_basis:
            # dict to matrix
            noise_matrix = np.zeros((len(noise), len(noise)))
            for idx in noise:
                noise_matrix[idx][idx] = noise[idx]
            # transform noise to original space
            proj_noise_matrix = VT.T @ noise_matrix
            noise = {idx: proj_noise_matrix[idx][idx] for idx in noise}

        # check for maximum noise
        for idx in noise:
            if idx not in noise_max:
                noise_max[idx] = 0.0
            noise_max[idx] = max(noise_max[idx], noise[idx])

    # print(f"Noise learned: {noise_max}")
    return noise_max

def privatize(output, learned_noise):
    """
    Add Anisotropic Gaussian noise to the output
    """
    for idx in range(len(output)):
        c = np.random.normal(0, scale=learned_noise[idx])
        output[idx] += c
    return output