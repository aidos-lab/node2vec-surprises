"""Main analysis script."""

import argparse
import glob
import os

import numpy as np

from metrics import diameter
from metrics import hausdorff_distance
from metrics import pairwise_function

import seaborn as sns
import matplotlib.pyplot as plt


def parse_filename(filename, normalise=True):
    """Parse filename into experiment description."""
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    tokens = basename.split('-')

    # Prepare data for experiments: store name of graphs etc. in there
    experiment = {
        'name': tokens[0]
    }

    # Ignore first and last one; we already know the first one, and we
    # don't need the last one since it only identifies the experiment.
    tokens = tokens[1:]
    tokens = tokens[:-1]

    for token in tokens:
        name = token[0]
        value = token[1:]

        if name == 'c':
            experiment['context'] = int(value)
        elif name == 'd':
            experiment['dimension'] = int(value)
        elif name == 'l':
            experiment['length'] = int(value)
        elif name == 'n':
            experiment['n_walks'] = int(value)
        elif name == 'p':
            experiment['edge_probability'] = float(value.replace('_', '.'))

    experiment['filename'] = filename
    experiment['data'] = np.loadtxt(filename, delimiter='\t')

    if normalise:
        diam = diameter(experiment['data'])
        experiment['data'] /= diam

    return experiment


def dist_parameters(exp1, exp2):
    """Distance between parameter sets."""
    dist = 0
    for (k1, v1), (k2, v2) in zip(exp1.items(), exp2.items()):
        assert k1 == k2
        if k1 == 'filename':
            continue
        else:
            dist += (v1 != v2)

    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file(s)', nargs='+')

    args = parser.parse_args()

    filenames = args.FILE
    experiments = [parse_filename(name) for name in filenames]

    H = pairwise_function(experiments, fn=hausdorff_distance, key='data')
    print(H)

    sns.heatmap(H)
    plt.show()
