import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm


class GCN(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, num_in_features)
        self.norm = GraphNorm(num_in_features)
        self.conv2 = GCNConv(num_in_features, num_out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GCNex(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.conv1 = GCNConv(num_in_features, num_in_features)
        self.norm = GraphNorm(num_in_features)
        self.conv2 = GCNConv(num_in_features, num_out_features)

    def forward(self, data, edge_index=None):
        # x, edge_index = data.x, data.edge_index
        print(type(data))
        if type(data) == torch.Tensor:
            x = data
        else:
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
