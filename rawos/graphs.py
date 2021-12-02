"""Simple random walk on graphs."""

import argparse
import collections

import numpy as np
import networkx as nx
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from rawos.random_walks import bound_chung
from rawos.random_walks import stationary_distribution
from rawos.random_walks import random_walk_simple
from rawos.random_walks import transition_matrix


def print_spectrum(A, name=None):
    """Print spectrum of matrix."""
    if name:
        print(name)

    if (A == A.T).all():
        print('symmetric')
        spec = np.linalg.eigvalsh(A)
    else:
        spec = np.linalg.eigvals(A)

    print(spec)
    print('\n')

    return spec


def bound(f, G, K=10):
    """Experiment with bound formulations."""
    degrees = np.asarray([d for n, d in G.degree()])

    pi = stationary_distribution(G)
    P = transition_matrix(G)

    A = nx.linalg.adjacency_matrix(G, weight=None).todense()
    D = np.diag(degrees)
    L = D - A
    I = np.eye(G.number_of_nodes())

    D_inv = np.diag(1.0 / degrees)

    for k in range(K):
        difference = (f - pi) @ np.linalg.matrix_power((I - D_inv @ L), k)
        print(np.linalg.norm(difference))


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
        start_distribution = np.ones(G.number_of_nodes())
        start_distribution /= G.number_of_nodes()
    else:
        start_distribution = np.zeros(G.number_of_nodes())
        start_distribution[args.start] = 1

    bound(start_distribution, G)
    raise 'heck'

    K = 50
    bounds = [bound_chung(G, k) for k in range(K)]
    pi = stationary_distribution(G)
    P = transition_matrix(G)

    # HIC SVNT LEONES
    #
    # TODO: formalise this somewhere...
    A = nx.linalg.adjacency_matrix(G, weight=None).todense()
    degrees = np.asarray([d for n, d in G.degree()])
    D_inv = np.diag(1.0 / degrees)
    D = np.diag(degrees)

    print_spectrum(A, 'A')
    spec_1 = print_spectrum(D_inv, 'D_inv')
    spec_2 = print_spectrum(A @ A.T, 'A @ A.T')

    Pt = P # np.linalg.matrix_power(P, 1)

    spec_3 = print_spectrum(Pt @ Pt.T, 'Pt @ Pt.T')

    print(np.linalg.matrix_power(np.sqrt(D) @ P @ np.sqrt(D_inv), 2))

    print(sorted(spec_1, reverse=True) * spec_2 * sorted(spec_1,
        reverse=True))

    raise 'heck'

    g = sns.lineplot(y=bounds, x=range(K))
    g.axhline(np.linalg.norm(pi))

    f = start_distribution

    diffs = [f @ np.linalg.matrix_power(P, k) for k in range(K)]
    diffs = [pi - x for x in diffs]
    diffs = [np.linalg.norm(x) for x in diffs]

    g = sns.lineplot(y=diffs, x=range(K))

    plt.show()



    hitting_times = collections.Counter(sorted(list(G.nodes())))

    for i in range(args.n):
        walk = random_walk_simple(G, args.length, start_distribution)
        hitting_times.update(walk)

    df = pd.DataFrame(list(hitting_times.items()))
    df = df.rename(
        columns={
            0: 'Node',
            1: 'Visits'
        }
    )

    g = sns.barplot(
        data=df,
        x='Node', y='Visits'
    )

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
