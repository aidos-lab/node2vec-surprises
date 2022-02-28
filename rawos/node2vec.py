"""Simple ``node2vec`` embedding generation."""

import argparse

import torch

import numpy as np

from torch_geometric.nn import Node2Vec
from torch_geometric.utils import erdos_renyi_graph


def main(args):
    num_nodes = 200
    edge_index = erdos_renyi_graph(num_nodes, 0.25)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        edge_index,
        embedding_dim=args.dimension,
        walk_length=args.length,
        context_size=args.context,
        walks_per_node=args.num_walks,
        num_negative_samples=1, p=1, q=1,
        sparse=True).to(device)

    loader = model.loader(batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        return z, torch.linalg.vector_norm(z)

    for epoch in range(1, 50):
        loss = train()
        z, norm = test()

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Norm: {norm:.4f}')

    np.savetxt(
        'embedding.csv',
        z.detach().cpu().numpy(),
        delimiter='\t',
        fmt='%.4f'
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--context', type=int, default=5)
    parser.add_argument('-d', '--dimension', type=int, default=32)
    parser.add_argument('-l', '--length', type=int, default=10)
    parser.add_argument('-n', '--num-walks', type=int, default=10)

    args = parser.parse_args()

    main(args)
