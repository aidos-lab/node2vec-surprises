"""Testing `scipy` implementation of Wasserstein distance."""

from scipy.stats import wasserstein_distance

import numpy as np
import ot

u = np.asarray([1.0, 2.0, 3.0], dtype=np.float)
v = u.copy()
np.random.default_rng().shuffle(v)

print(u, v)
print(wasserstein_distance(u, v))
print(wasserstein_distance(np.arange(len(u)), np.arange(len(v)), u, v))

M = np.ones((len(u), len(v)))
M[np.diag_indices_from(M)] = 0.0

print(ot.emd2(u, v, M))
