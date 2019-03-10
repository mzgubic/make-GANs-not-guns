import numpy as np

def triangular1D(N):
    return np.random.triangular(0, 1, 1, N).reshape(-1, 1)

def uniform1D(N):
    return np.random.uniform(0, 1, N).reshape(-1, 1)

def beta1D(N):
    return np.random.beta(2, 3, N).reshape(-1, 1)

def input_noise(noise_dim=3):
    def f(N):
        return np.random.normal(0, 1, size=(N, noise_dim))
    return f

