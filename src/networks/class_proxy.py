import torch
import torch.nn as nn

from torch_geometric.data import Data

class ProxyHead(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features):
        return self.layers(features)

def graphgen(node_embeddings):
    device = node_embeddings.device
    _, num_nodes, _ = node_embeddings.shape
    sims = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
    sims = sims * torch.ones(num_nodes, num_nodes).fill_diagonal_(0).to(device)  # disregard self-similarities
    directed: torch.Tensor = sims > (sims.sum(dim=2) / num_nodes - 1).unsqueeze(dim=2)  # average only over non-zero elms
    undirected = directed + directed.transpose(1, 2)
    assert torch.all(undirected == undirected.transpose(1, 2)).item()  # validate symmetrization
    edges = undirected.nonzero()

    edge_lists = []
    offset = 0
    graphs = []
    for i, sample in enumerate(undirected):
        num_edges = undirected[i].sum()
        edges_i = edges[offset: offset + num_edges]
        # Edge list in COO format
        edges_i = edges_i[:, 1:].T
        edge_lists.append(edges_i)
        offset = offset + num_edges
        graphs.append(Data(x = node_embeddings[i], edge_index=edges_i))

    return graphs

def create_proxy_graphs(proxy_heads):
    node_embeddings = torch.cat([proxy_head.layers[-1].weight.unsqueeze(0) for proxy_head in proxy_heads]).transpose(0, 1)
    proxy_graphs = graphgen(node_embeddings)
    return proxy_graphs, node_embeddings
