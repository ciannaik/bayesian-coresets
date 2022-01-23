import numpy as np

np.random.seed(1)
N = 10000
D = 50
mu = 10*np.random.randn(D)
sig = 10.
np.save("synth_gauss.npy", mu + sig*np.random.randn(N, D))
