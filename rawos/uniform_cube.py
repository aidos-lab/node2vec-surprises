import numpy as np

from sklearn.metrics import pairwise_distances


values = []

for i in range(10):
    X = np.random.uniform(size=(200, 16))
    D = pairwise_distances(X)

    values.append(D.mean())

print(np.mean(values), np.std(values))
