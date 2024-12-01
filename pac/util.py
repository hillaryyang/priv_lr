import numpy as np

def get_samples_safe(X, y, n_samples):
    # randomly sample with index
    idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]

    return X_sample, y_sample