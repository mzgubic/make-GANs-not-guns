import numpy as np

def triangular1D(N):
    return np.random.triangular(-1, 1, 1, N).reshape(-1, 1)

def input_noise(N, noise_dim=3):
    return np.random.normal(0, 1, size=(N, noise_dim))

