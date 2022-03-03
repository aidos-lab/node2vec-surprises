"""Metrics and pseudo-metrics for comparing point clouds."""

import numpy as np
import ot

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon


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

    # Check whether dimensions are compatible.
    if X.shape[1] != Y.shape[1]:
        return np.nan

    distances = pairwise_distances(X=X, Y=Y)

    d_XY = np.max(np.min(distances, axis=1))
    d_YX = np.max(np.min(distances, axis=0))

    return max(d_XY, d_YX)


def jensenshannon_distance(X, Y):
    """Computes the Jensen-Shannon distance of two embeddings of the same number of points"""
    
    diamx = diameter(X)
    X = X/diamx
    
    diamy = diameter(Y)
    Y = Y/diamy
    
    dX = pairwise_distances(X, metric='euclidean')
    dY = pairwise_distances(Y, metric='euclidean')

    DX = []
    DY = []
    for i in range(X.shape[0]): 
        for j in range(i+1, X.shape[0]): 
            DX.append(dX[i,j])

    for i in range(Y.shape[0]): 
        for j in range(i+1, Y.shape[0]): 
            DY.append(dY[i,j])

    DistX, binsX = np.histogram(DX, bins=100)
    DistY, binsY = np.histogram(DY, bins=100)

    DistX = DistX.astype(np.float)
    DistY = DistY.astype(np.float)

    DistX /= DistX.sum()
    DistY /= DistY.sum()

    ## uses base e, upper bound of JSD is ln(2)
    ## in base b upper boud of JSD is log_b(2)
    return  jensenshannon(DistY, DistX, base=None) 


def wasserstein_distance(X, Y, metric='euclidean'):
    """Calculate Wasserstein distance between point clouds.

    Calculates the Wasserstein distance between two finite metric
    spaces, i.e. two finite point clouds, using ``metric`` as its
    base metric.
    """
    M = ot.dist(X, Y, metric=metric)
    M /= M.max()

    # Uniform weights for all samples.
    a = np.ones((len(X), ))
    b = np.ones((len(Y), ))

    dist = ot.emd2(a, b, M)
    return dist


def pairwise_function(X, fn, Y=None, key=None):
    """Pairwise scalar value calculation with an arbitrary function."""
    n = len(X)
    m = len(X) if Y is None else len(Y)
    D = np.empty((n, m), dtype=float)

    if Y is None:
        Y = X

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if key is None:
                D[i, j] = fn(x, y)
            else:
                D[i, j] = fn(x[key], y[key])

    return D
