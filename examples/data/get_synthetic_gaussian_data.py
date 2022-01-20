import numpy as np

np.random.seed(1)
N = 1000000
D = 100
mu = 5.
sig = 1.
np.save("synth_gauss.npy", mu + sig*np.random.randn(N, D))
