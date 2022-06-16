"""Make gallery of embeddings."""

import argparse

from analyse_results import assign_groups
from analyse_results import parse_filename

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file(s)', nargs='+')

    args = parser.parse_args()

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

    n_rows = n_groups
    n_cols = len([e for e in experiments if e['group'] == 0])

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=False,
        sharey=False,
        figsize=(8, 8)
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)

    for group in range(n_groups):
        experiments_group = [e for e in experiments if e['group'] == group]

        row = group
        for col, exp in enumerate(experiments_group):
            X = PCA(n_components=2).fit_transform(exp['data'])

            ax = axes[row, col]
            ax.set_visible(True)
            ax.scatter(X[:, 0], X[:, 1], s=1.0)

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.savefig('/tmp/node2vec_gallery.png')
    plt.show()
