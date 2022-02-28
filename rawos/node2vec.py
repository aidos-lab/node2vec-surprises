"""Simple ``node2vec`` embedding generation."""

import argparse

import os.path as osp

import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import erdos_renyi_graph


def main(args):
    num_nodes = 200
    edge_index = erdos_renyi_graph(num_nodes, 0.25)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index, embedding_dim=32, walk_length=10,
                     context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

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
        return torch.linalg.vector_norm(z)

    for epoch in range(1, 50):
        loss = train()
        norm = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Norm: {norm:.4f}')

    @torch.no_grad()
    def plot_points():
        model.eval()
        z = model(torch.arange(num_nodes, device=device))
        z = PCA(n_components=2).fit_transform(z.cpu().numpy())

        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, 0], z[:, 1], s=20)
        plt.axis('off')
        plt.show()

    plot_points()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
