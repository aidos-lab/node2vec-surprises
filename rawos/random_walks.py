"""Random walk functors on structures."""

import numpy as np


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
