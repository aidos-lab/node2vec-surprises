"""Main analysis script."""

import argparse
import glob
import os


def parse_filename(filename):
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
    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', type=str, help='Input directory')

    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.DIRECTORY, '*.tsv'))
    experiments = [parse_filename(name) for name in filenames]
