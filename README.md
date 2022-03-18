# RAWOS: Random Walks On Structures

Running the code (example):

    # Let's run for 50 epochs, generate 16-dimensional embeddings,
    # and keep everything at default parameters.
    $ python node2vec.py -d 16 -e 50

## Examples

### Creating a gallery of embeddings

    # You can select different point clouds to visualise here. The
    # gallery script is sufficiently smart to enlarge its grid.
    $ python gallery.py ../results/lm/*-d64*.tsv
