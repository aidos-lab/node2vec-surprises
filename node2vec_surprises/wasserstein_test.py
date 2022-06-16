"""Testing `scipy` implementation of Wasserstein distance."""

from scipy.stats import entropy
from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance

from scipy.spatial.distance import jensenshannon

import numpy as np
import ot

u = np.asarray([0.1, 0.2, 0.7], dtype=float)
v = np.asarray([0.2, 0.1, 0.7], dtype=float)
n = len(u)
x = np.arange(n, dtype=float)

print(u, v)
print('Entropy:', entropy(u, v))
print('Energy distance:', energy_distance(u, v))
print('Jensen--Shannon distance:', jensenshannon(u, v))
print('Wasserstein distance:', wasserstein_distance(u, v))

M = ot.dist(x.reshape(n, 1), x.reshape(n, 1), metric='minkowski')
M /= M.max()

print('EMD:', ot.emd2(u, v, M))
