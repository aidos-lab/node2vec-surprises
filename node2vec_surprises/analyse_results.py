"""Main analysis script."""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

import scipy.stats

from metrics import diameter
from metrics import hausdorff_distance
from metrics import jensenshannon_distance
from metrics import link_auc
from metrics import link_distributions_emd
from metrics import mean_distance
from metrics import persistent_entropy_point_cloud
from metrics import total_persistence_point_cloud
from metrics import wasserstein_distance

from metrics import pairwise_function
from metrics import summary_statistic_function

import seaborn as sns
import matplotlib.pyplot as plt

# Maps a function name to an actual `callable`, thus enabling us to
# configure this via `argparse`. The second entry indicates whether
# a measure permits pairwise comparisons.
fn_map = {
    'hausdorff': (hausdorff_distance, True),
    'js': (jensenshannon_distance, True),
    'link_auc': (link_auc, False),
    'link_distributions': (link_distributions_emd, False),
    'mean_distance': (mean_distance, False),
    'persistent_entropy': (persistent_entropy_point_cloud, False),
    'total_persistence': (total_persistence_point_cloud, False),
    'wasserstein': (wasserstein_distance, True)
}


# Maps a function name to a nice label that will be used for plotting.
label_map = {
    'hausdorff': 'Hausdorff distance',
    'link_auc': 'Link reconstruction (AUPRC)',
    'link_distributions': 'Link distribution ($W_q$)',
    'wasserstein': 'Wasserstein distance',
}


def make_labels(df):
    """Create labels for captioning a data frame."""
    def _row_to_label(row):
        key_map = {
            'length': '$L$',
            'n_walks': '$N$',
            'context': '$C$',
            'q': '$q$',
        }

        row_dict = dict(row)
        var_dict = dict()

        keys = sorted(row_dict.keys())

        for k in keys:
            if k in key_map:
                var_dict[key_map[k]] = row_dict.pop(k)

        label = ''
        for k, v in var_dict.items():
            label += f'{k} = {v}, '

        label = label[:-2]
        return row['group'], label

    labels = df.apply(_row_to_label, axis='columns')
    return labels


def parse_filename(filename, normalise=True):
    """Parse filename into experiment description."""
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    tokens = basename.split('-')

    # Prepare data for experiments: store name of graphs etc. in there
    experiment = {
        'name': tokens[0],
        'length': 0,
        'n_walks': 0,
        'context': 0,
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
        elif name == 'q':
            experiment['q'] = int(value)
        elif token == 'keep':
            experiment['keep'] = True

    experiment['filename'] = filename
    experiment['data'] = np.loadtxt(filename, delimiter='\t')

    if 'q' not in experiment:
        experiment['q'] = 1

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


def analyse_distances(args, experiments, n_groups):
    """Analyse and visualise inter- and intra-group distances."""
    data = []

    for x, y in itertools.combinations_with_replacement(range(n_groups), 2):
        X = [e for e in experiments if e['group'] == x]
        Y = [e for e in experiments if e['group'] == y]

        if x == y:
            Y = None

        distances = pairwise_function(
            X, stats_fn, Y=Y, key='data'
        )

        data.append(pd.DataFrame.from_dict({
            'group1': x,
            'group2': y,
            'distances':  distances[distances > 0.0].ravel().tolist()
        }))

    df = pd.concat(data)
    print(df)

    for name, df_ in df.groupby(['group1', 'group2']):
        sns.kdeplot(
            data=df_,
            x='distances',
            label=name,
            color='k' if name[0] == name[1] else 'b',
            common_norm=True,
            fill=True,
            alpha=0.1,
        )

    plt.show()


def analyse_statistics(args, experiments, n_groups):
    """Analyse and visualise statistics."""
    # Data frame with stats; will be visualised later on.
    df = []

    A = np.loadtxt(args.adjacency) if args.adjacency is not None else None

    for group in range(n_groups):
        experiments_group = [e for e in experiments if e['group'] == group]

        # Pairwise comparisons are possible between members of each
        # group.
        if pairwise:
            distances_in_group = pairwise_function(
                experiments_group, fn=stats_fn, key='data', A=A
            )

            distances_in_group = distances_in_group.ravel()
            distances_in_group = distances_in_group[distances_in_group > 0.0]

            row = {'distances': distances_in_group.tolist()}

        # We can only extract statistics for group members.
        else:
            stats = summary_statistic_function(
                experiments_group, fn=stats_fn, key='data', A=A
            )

            stats = stats.squeeze()

            if not stats.shape:
                stats = np.expand_dims(stats, axis=0)

            # Assume that we are getting more than one value, such as
            # a total persistence value for each dimension.
            if len(stats.shape) > 1:
                stats_dimensions = []
                for d, _ in enumerate(stats.T):
                    stats_dimensions.extend([d] * len(stats))

                row = {
                    'stats': stats.T.ravel().tolist(),
                    'stats_dimension': stats_dimensions
                }
            else:
                row = {'stats': stats.tolist()}

        parameters = get_variable_parameters(
            experiments_group[0],
            as_string=False
        )

        row.update(parameters)

        df.append(pd.DataFrame.from_dict(row))

    df = pd.concat(df)
    df['group'] = df['group'].astype('category')

    attribute = 'distances' if 'distances' in df.columns else 'stats'

    if args.no_labels:
        plt.figure(figsize=(16, 3))
    else:
        plt.figure(figsize=(16, 6))

    sns.set_palette('Set2')

    if 'stats_dimension' in df.columns:
        g = sns.FacetGrid(
            data=df,
            row='stats_dimension',
            sharey=False, sharex=True
        )
        g.map_dataframe(
            sns.boxplot,
            x='group',
            y='stats',
            hue=args.hue,
            dodge=False,
        )
        g.add_legend()
    else:
        g = sns.boxplot(
            data=df,
            x=df['group'], y=attribute,
            hue=args.hue,
            dodge=False
        )

        labels = make_labels(df)
        labels = sorted(set(labels), key=lambda x: x[0])
        labels = list(map(lambda x: x[1], labels))

        if args.function in label_map:
            plt.ylabel(label_map[args.function])

        sns.despine()

        g.set_xticks(range(len(labels)))

        if args.no_labels:
            g.set_xticklabels([""] * len(labels))
        else:
            g.set_xticklabels(labels)
            plt.subplots_adjust(bottom=0.5)

        plt.xticks(rotation=90)

    if args.output is not None:
        plt.savefig(args.output)

    # Can only analyse statistical significance if we are dealing with
    # distances.
    if attribute == 'distances':
        df_per_group = {
            name: col for name, col in df.groupby('group')['distances']
        }

        P = np.zeros((n_groups, n_groups))

        for g1, x in df_per_group.items():
            for g2, y in df_per_group.items():
                if g1 <= g2:
                    continue

                test = scipy.stats.wilcoxon(x, y)
                P[g1, g2] = test.pvalue

        P = 0.5 * (P + P.T)
        P = P < 0.05 / (0.5 * n_groups * (n_groups - 1))

        print(
            'Percentage of statistically significantly different '
            'distributions:', 100 * np.sum(P) / (P.shape[0] ** 2)
        )

        plt.figure()
        sns.heatmap(P, vmin=0, vmax=1.0, cmap='RdYlGn')

    plt.tight_layout()

    if args.output is None:
        plt.show()


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

    parser.add_argument(
        '-d', '--distances',
        action='store_true',
        help='If set, analyses inter- and intra-group distances. Only works '
             'if the selected function permits pairwise comparisons.'
    )

    parser.add_argument(
        '-a', '--adjacency',
        type=str,
        help='If set, loads adjacency matrix from specified path. This will '
              'be required for link distribution analysis.'
    )

    parser.add_argument(
        '-n', '--no-labels',
        action='store_true',
        help='If set, suppresses label output'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        type=str,
        help='If set, stores plots in specified filename.'
    )

    args = parser.parse_args()

    stats_fn = fn_map[args.function][0]
    pairwise = fn_map[args.function][1]

    if args.distances:
        assert pairwise

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

    # Determine which mode to follow here. We are either analysing
    # distances or visualising statistics.

    if args.distances:
        analyse_distances(args, experiments, n_groups)
    else:
        analyse_statistics(args, experiments, n_groups)
