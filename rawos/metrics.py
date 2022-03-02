"""Metrics and pseudo-metrics for comparing point clouds."""

import numpy as np

from sklearn.metrics import pairwise_distances


def diameter(X, metric='euclidean'):
    """Calculate diameter of a point cloud."""
    distances = pairwise_distances(X, metric=metric)
    return np.max(distances)


def hausdorff_distance(X, Y, metric='euclidean'):
    """Calculate Hausdorff distance between point clouds.

    Calculates the Hausdorff distance between two finite metric spaces,
    i.e. two finite point clouds.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    distances = pairwise_distances(X=X, Y=Y)

    d_XY = np.max(np.min(distances, axis=1))
    d_YX = np.max(np.min(distances, axis=0))

    return max(d_XY, d_YX)


def pairwise_function(X, fn, Y=None):
    """Pairwise scalar value calculation with an arbitrary function."""
    n = len(X)
    m = len(X) if Y is None else len(Y)
    D = np.empty((n, m), dtype=float)

    if Y is None:
        Y = X

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            D[i, j] = fn(x, y)

    return D
