"""Simple random walk on graphs."""

import argparse
import collections

import numpy as np
import networkx as nx

from rawos.random_walks import random_walk_simple


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--length',
        type=int,
        default=3,
        help='Length of random walk'
    )
    parser.add_argument(
        '-n',
        type=int,
        default=100,
        help='Number of random walks',
    )
    parser.add_argument(
        '-s', '--start',
        type=int,
        default=None,
        help='Start node (to simulate a Dirac measure)'
    )

    args = parser.parse_args()

    G = nx.generators.social.les_miserables_graph()

    if args.start is None:
        degrees = np.asarray(list(dict(G.degree()).values()))
        start_distribution = degrees / np.sum(degrees)

    hitting_times = collections.Counter(list(G.nodes()))

    for i in range(args.n):
        walk = random_walk_simple(G, args.length, start_distribution)
        hitting_times.update(walk)
