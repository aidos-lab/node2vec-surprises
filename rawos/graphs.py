"""Simple random walk on graphs."""

import argparse
import collections

import numpy as np
import networkx as nx
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from rawos.random_walks import bound_chung
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

    bounds = [bound_chung(G, k) for k in range(10)]
    sns.lineplot(y=bounds, x=range(10))

    plt.show()

    if args.start is None:
        degrees = np.asarray(list(dict(G.degree()).values()))
        start_distribution = degrees / np.sum(degrees)

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
