"""Create random embedding for a distance matrix."""

import argparse
import sys

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', help='Input file')
    parser.add_argument(
        '-d', '--dimension',
        default=16,
        type=int,
        help='Embedding dimension'
    )

    args = parser.parse_args()

    A = np.loadtxt(args.FILE)
    n = len(A)
    d = args.dimension

    # TODO: Could also do Laplacian embedding here; might be a good
    # baseline comparison.
    rng = np.random.default_rng()
    X = rng.normal(size=(n, d))

    np.savetxt(sys.stdout, X, fmt='%.4f')
