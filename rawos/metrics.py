"""Metrics and pseudo-metrics for comparing point clouds."""

import numpy as np

import ot

from gtda.homology import VietorisRipsPersistence

from sklearn.metrics import auc
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

from scipy.spatial.distance import jensenshannon


def diameter(X, metric='euclidean', **kwargs):
    """Calculate diameter of a point cloud."""
    distances = pairwise_distances(X, metric=metric)
    return np.max(distances)


def hausdorff_distance(X, Y, metric='euclidean', **kwargs):
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


def link_distributions_emd(X, A=None, **kwargs):
    """Evaluate Earth Mover's Distance between link distributions."""
    if A is None:
        return np.inf

    n = A.shape[0]
    P_original = A / (0.5 * np.sum(A))
    P_original = P_original[np.triu_indices(n, k=1)]

    # Create empirical 'observed' link distribution and normalise it
    # afterwards. Some statistical distances require normalised data
    # so this ensures overall validity.
    P_observed = 1 / (1 + np.exp(-pairwise_kernels(X)))
    P_observed = P_observed[np.triu_indices(n, k=1)]
    P_observed /= np.sum(P_observed)

    n = len(P_original)
    x = np.arange(n, dtype=float)

    metric = kwargs.get('metric', 'minkowski')
    M = ot.dist(
        x.reshape(n, 1),
        x.reshape(n, 1),
        metric=metric
    )
    M /= M.max()

    dist = ot.emd2(P_original, P_observed, M)
    return dist 


def link_auc(X, A=None, **kwargs):
    """Evaluate AUC for edge prediction."""
    if A is None:
        return np.inf

    n = A.shape[0]
    K = pairwise_kernels(X)

    # Count edges and non-edges (don't use symmetry or anything else).
    n_edges = A.sum()
    n_non_edges = n**2 - n_edges

    thresholds = list(K.ravel())

    tpr = []
    fpr = []
    thr = []

    for threshold in np.linspace(np.min(thresholds), np.max(thresholds), 100):
        E = (K >= threshold)
        E = E.astype(np.int)

        # Makes it possible to detect true negatives as well
        E *= 2

        # 2 - 1 ==  1: TP
        # 2 - 0 ==  2: FP
        # 0 - 0 ==  0: TN
        # 0 - 1 == -1: FN
        diff = E - A

        tp = (diff == +1).sum()
        fp = (diff == +2).sum()
        tn = (diff == +0).sum()
        fn = (diff == -1).sum()

        tpr.append(tp / n_edges)
        fpr.append(fp / n_non_edges)
        thr.append(threshold)

    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    return auc(fpr, tpr) 


def jensenshannon_distance(X, Y, **kwargs):
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


def wasserstein_distance(X, Y, metric='euclidean', **kwargs):
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


def get_dimension(diagram, dim=0):
    """Get specific dimension of a persistence diagram."""
    mask = diagram[..., 2] == dim
    diagram = diagram[mask][:, :2]
    return diagram


def total_persistence(diagram, dim=0, p=2, **kwargs):
    """Calculate total persistence of a persistence diagram."""
    diagram = get_dimension(diagram, dim)

    persistence = np.diff(diagram)
    persistence = persistence[np.isfinite(persistence)]

    # Ensure that the simplification is only performed over a given
    # axis.
    result = np.sum(np.power(np.abs(persistence), p), axis=0)
    return result


def total_persistence_point_cloud(X, max_dim=1, **kwargs):
    """Calculate total persistence values of a point cloud."""
    ph = VietorisRipsPersistence(
        metric='euclidean',
        homology_dimensions=tuple(range(max_dim + 1)),
        infinity_values=None
    ).fit_transform([X])

    diagrams = ph[0]

    total_pers = []
    for i in range(max_dim + 1):
        total_pers.append(total_persistence(diagrams, dim=i))

    return np.asarray(total_pers)


def mean_distance(X, metric='euclidean', **kwargs):
    D = pairwise_distances(X, metric=metric)
    return np.mean(D)


def persistent_entropy(diagram, dim=0, **kwargs):
    """Calculate persistent entropy of a diagram."""
    diagrams = get_dimension(diagram, dim)

    persistence = np.diff(diagram)
    persistence = persistence[np.isfinite(persistence)]

    persistence_sum = np.sum(persistence)
    probabilities = persistence / persistence_sum

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    log_prob = np.log2(probabilities,
                       out=np.zeros_like(probabilities),
                       where=(probabilities > 0))

    return np.sum(-probabilities * log_prob)


def persistent_entropy_point_cloud(X, max_dim=1, **kwargs):
    """Calculate persistent entropy values of a point cloud."""
    ph = VietorisRipsPersistence(
        metric='euclidean',
        homology_dimensions=tuple(range(max_dim + 1)),
        infinity_values=None
    ).fit_transform([X])

    diagrams = ph[0]

    entropies = []
    for i in range(max_dim + 1):
        entropies.append(persistent_entropy(diagrams, dim=i))

    return np.asarray(entropies)


def pairwise_function(X, fn, Y=None, key=None, **kwargs):
    """Pairwise scalar value calculation with an arbitrary function."""
    n = len(X)
    m = len(X) if Y is None else len(Y)
    D = np.empty((n, m), dtype=float)

    if Y is None:
        Y = X

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if key is None:
                D[i, j] = fn(x, y, **kwargs)
            else:
                D[i, j] = fn(x[key], y[key], **kwargs)

    return D


def summary_statistic_function(X, fn, key=None, **kwargs):
    """Scalar value calculation with an arbitrary function."""
    values = []

    for x in X:
        if key is None:
            values.append(fn(x, **kwargs))
        else:
            values.append(fn(x[key], **kwargs))

    return np.asarray(values)
