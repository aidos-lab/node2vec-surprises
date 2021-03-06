# On the Surprising Behaviour of `node2vec`

![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/node2vec-surprises) ![GitHub](https://img.shields.io/github/license/aidos-lab/node2vec-surprises)

This is the code to our paper on the [ICML 2022 Workshop on Topology,
Algebra, and Geometry in Machine Learning]:

```bibtex
@unpublished{Hacker22a,
  title         = {On the Surprising Behaviour of \texttt{node2vec}},
  author        = {Celia Hacker and Bastian Rieck},
  year          = {2022},
  eprint        = {2206.08252},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  type          = {Preprint},
  repository    = {https://github.com/aidos-lab/node2vec-surprises},
}
```

## Installation

We suggest using [`poetry`](https://python-poetry.org) to install the
code:

```bash
$ poetry install
```

We have tested this repository with Python 3.9 and Python 3.8.1.

## Examples

### Creating some embeddings

Running the code (example):

    # Let's run for 50 epochs, generate 16-dimensional embeddings,
    # and keep everything at default parameters.
    $ python node2vec.py -d 16 -e 50

### Creating a gallery of embeddings

    # You can select different point clouds to visualise here. The
    # gallery script is sufficiently smart to enlarge its grid.
    $ python gallery.py ../results/lm/*-d64*.tsv

### Analysing distributions

    # The script is smart enough to check whether pairwise distances
    # can be calculated and compared here.
    $ python analyse_results.py ../results/lm/*.tsv  --hue dimension --function mean_distance
    $ python analyse_results.py ../results/lm/*.tsv  --hue dimension --function hausdorff
    $ python analyse_results.py ../results/lm/*.tsv  --hue dimension --function wasserstein

Alternatively, we can also visualise kernel density estimates of
intra-group and inter-group distances:

    $ python analyse_results.py ../results/lm/*-d64*.tsv  --hue dimension --function wasserstein -d

Note that this only works for pairwise distance metrics such as the
Wasserstein distance.

## Quality assessment

To perform a rudimentary quality assessment analysis, you need to
provide a set of embeddings as well as an adjacency matrix to the
analysis script:

    $ python analyse_results.py ../results/lm/*.tsv  --hue dimension --function link_distributions --adjacency ../results/lm/A.txt
