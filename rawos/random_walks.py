"""Random walk functors on structures."""

import numpy as np
import networkx as nx


def stationary_distribution(G):
    """Return stationary distribution for graph."""
    E = G.number_of_edges()
    degrees = np.asarray([d for n, d in G.degree()])

    return degrees / (2 * E)


def transition_matrix(G):
    """Return transition matrix of graph."""
    degrees = np.asarray([d for n, d in G.degree()])
    D_inv = np.diag(1.0 / degrees)

    return D_inv @ nx.linalg.adjacency_matrix(G, weight=None)


def bound_chung(G, k):
    """Calculate bound provided by Fan Chung in 'Spectral Graph Theory'.

    Parameters
    ----------
    G : `nx.Graph`
        Input graph

    k : int
        Number of steps of random walk

    Returns
    -------
    Upper bound for the distance between *any* initial distribution and
    the stationary distribution of `P` after `k` steps.
    """
    degrees = list(dict(G.degree()).values())
    d = np.min(degrees)
    D = np.max(degrees)

    spectrum = sorted(nx.normalized_laplacian_spectrum(G, weight=None))
    upper = spectrum[-1]
    lower = spectrum[1]

    lam = lower if 1 - lower >= upper - 1 else 2 - upper
    return np.exp(-k * lam) * np.sqrt(D / d)


def random_walk_simple(G, length, start_distribution):
    """Perform simple random walk of a certain length on a graph."""
    v = np.random.choice(G.nodes(), p=start_distribution)
    walk = [v]

    for i in range(length):
        neighbours = list(G.neighbors(v))
        degrees = np.asarray([
            G.degree(nb) for nb in neighbours
        ])

        v = np.random.choice(neighbours, p=degrees / np.sum(degrees))
        walk.append(v)

    return walk
