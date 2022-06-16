"""Create adjacency matrix of well-known graphs."""

import argparse

import networkx as nx
import numpy as np


def lm():
    G = nx.generators.les_miserables_graph()
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'NAME',
        type=str,
        help='Name of graph to generate'
    )

    args = parser.parse_args()

    if args.NAME == 'lm':
        G = lm()

    A = nx.adjacency_matrix(G, weight=None).toarray()
    np.savetxt('/tmp/A.txt', A, fmt='%d')
