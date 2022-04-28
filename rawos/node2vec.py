"""Simple ``node2vec`` embedding generation."""

import argparse

import torch
import uuid

import numpy as np

import pytorch_lightning as pl

from torch_geometric.nn import Node2Vec
from torch_geometric.utils.convert import from_networkx

from networkx import adjacency_matrix
from networkx import stochastic_block_model

from networkx.generators import les_miserables_graph

edge_index = None
num_nodes = 0


def sbm(m=2, n=100):
    """Return SBM graph with ``m`` groups and ``m * n`` nodes."""
    p = 0.8
    q = 0.2

    if m == 2:
        probs = [[p, q], [q, p]]
        N = [n, n]
    elif m == 3:
        probs = [[p, q, q], [q, p, q], [q, q, p]]
        N = [n, n, n]
    else:
        raise RuntimeError('Unexpected number of groups.')

    # Note that the seed is fixed; we want to control for *this* source of
    # randomness at least.
    sbm = stochastic_block_model(N, probs, sparse=True, seed=42)
    return sbm


class node2vec(pl.LightningModule):
    """`node2vec` module."""

    def __init__(self, args):
        """Create new `node2vec` module.

        Parameters
        ----------
        args
            Command-line arguments, parsed using `argparse`.
        """
        super().__init__()

        global edge_index
        global num_nodes

        if edge_index is None:
            if args.graph == 'lm':
                G = les_miserables_graph()
            elif args.graph == 'sbm2':
                G = sbm(2)
            elif args.graph == 'sbm3':
                G = sbm(3)

            self.A = adjacency_matrix(G, weight=None).toarray()

            edge_index = from_networkx(G).edge_index
            num_nodes = G.number_of_nodes()

        self.model = Node2Vec(
            edge_index,
            embedding_dim=args.dimension,
            walk_length=args.length,
            context_size=args.context,
            walks_per_node=args.num_walks,
            num_negative_samples=1,
            p=1,
            q=args.q,
            sparse=True
        )

    def configure_optimizers(self):
        opt = torch.optim.SparseAdam(self.model.parameters(), lr=1e-2)
        return opt

    def forward(self, x):
        embedding = self.model(x)

    def training_step(self, train_batch, batch_idx):
        pos_rw, neg_rw = train_batch
        loss = self.model.loss(pos_rw, neg_rw)

        self.log('train_loss', loss)
        return loss

    def get_embedding(self):
        embedding = self.model()
        return embedding.detach().numpy()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        n_gpus = 1
    else:
        n_gpus = 0

    model = node2vec(args)

    train_loader = model.model.loader(
        batch_size=num_nodes,
        shuffle=True,
        num_workers=4
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='train_loss',
        patience=10,
    )

    # FIXME: Currently, we are never using the GPU because the detection
    # routine is not sufficiently smart to detect cases where we have
    # CUDA but not available GPU.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=0,
        callbacks=early_stopping,
    )
    trainer.fit(model, train_loader)

    model.eval()
    z = model.get_embedding()

    id_ = str(uuid.uuid4().hex)

    filename = 'lm'
    filename += f'-c{args.context}'
    filename += f'-d{args.dimension}'
    filename += f'-l{args.length}'
    filename += f'-n{args.num_walks}'
    filename += f'-q{args.q}'

    filename += f'-{id_}'
    filename += '.tsv'

    np.savetxt(
        filename,
        z,
        delimiter='\t',
        fmt='%.4f'
    )

    np.savetxt(f'A-{args.graph}.txt', model.A, fmt='%d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--context', type=int, default=5)
    parser.add_argument('-d', '--dimension', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--length', type=int, default=10)
    parser.add_argument('-n', '--num-walks', type=int, default=10)

    parser.add_argument('-q', type=int, default=1)

    parser.add_argument('-N', '--num-runs', type=int, default=10)
    parser.add_argument(
        '-g', '--graph',
        type=str,
        default='lm',
        choices=['lm', 'sbm2', 'sbm3']
    )

    args = parser.parse_args()

    for i in range(args.num_runs):
        main(args)
