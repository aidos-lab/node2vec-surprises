"""Main analysis script."""

import argparse
import json
import os

import numpy as np
import pandas as pd

import scipy.stats as stats

from metrics import diameter
from metrics import hausdorff_distance
from metrics import pairwise_function
from metrics import wasserstein_distance

import seaborn as sns
import matplotlib.pyplot as plt

# Maps a function name to an actual `callable`, thus enabling us to
# configure this via `argparse`.
fn_map = {
    'hausdorff': hausdorff_distance,
    'wasserstein': wasserstein_distance
}

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

        # By default, the graph is modified instead of kept when
        # repeating the experiment.
        experiment['keep'] = False

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
        elif token == 'keep':
            experiment['keep'] = True

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


def get_variable_parameters(experiment, as_string=True):
    """Return array of all variable parameters."""
    ignored = [
        'filename',
        'data',
    ]

    if as_string:
        parameters = [
            f'{k} = {v}' for k, v in experiment.items() if k not in ignored
        ]
    else:
        parameters = {
            k: v for k, v in experiment.items() if k not in ignored
        }

    return parameters


def assign_groups(experiments):
    """Assign (arbitrary) groups to experiments based on parameters."""
    parameters = list(map(get_variable_parameters, experiments))
    parameters = [
        ', '.join(p) for p in parameters
    ]

    unique_parameters = dict.fromkeys(parameters)
    unique_parameters = dict({
            p: i for i, p in enumerate(unique_parameters)
    })

    for e, p in zip(experiments, parameters):
        e['group'] = unique_parameters[p]

    print(json.dumps(unique_parameters, indent=2))
    return experiments, len(unique_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file(s)', nargs='+')
    parser.add_argument(
        '--hue',
        type=str,
        default=None,
        help='Attribute by which to colour results.'
    )
    parser.add_argument(
        '-f', '--function',
        type=str,
        default='hausdorff',
        help='Select pairwise analysis function'
    )

    args = parser.parse_args()

    pairwise_fn = fn_map[args.function]

    filenames = args.FILE
    experiments = [parse_filename(name) for name in filenames]

    experiments = sorted(
        experiments,
        key=lambda x: (
            x['dimension'],
            x['length'],
            x['n_walks'],
            x['context'])
    )

    experiments, n_groups = assign_groups(experiments)

    # Data frame with stats; will be visualised later on.
    df = []

    for group in range(n_groups):
        experiments_group = [e for e in experiments if e['group'] == group]
        distances_in_group = pairwise_function(
            experiments_group, fn=pairwise_fn, key='data'
        )

        distances_in_group = distances_in_group.ravel()
        distances_in_group = distances_in_group[distances_in_group > 0.0]

        parameters = get_variable_parameters(
            experiments_group[0],
            as_string=False
        )

        row = {'distances': distances_in_group.tolist()}
        row.update(parameters)

        df.append(pd.DataFrame.from_dict(row))

    df = pd.concat(df)
    df = df.astype({'group': 'int32'})

    sns.boxplot(
        data=df,
        x=df['group'], y='distances',
        hue=args.hue,
        dodge=False
    )

    df_per_group = {
        name: col for name, col in df.groupby('group')['distances']
    }

    P = np.zeros((n_groups, n_groups))

    for g1, x in df_per_group.items():
        for g2, y in df_per_group.items():
            if g1 <= g2:
                continue

            test = stats.wilcoxon(x, y)
            P[g1, g2] = test.pvalue

    P = 0.5 * (P + P.T)
    P = P < 0.05 / (n_groups**2)

    fig = plt.figure()
    sns.heatmap(P, vmin=0, vmax=1.0, cmap='RdYlGn')

    plt.tight_layout()
    plt.show()
