import torch
from torch import nn


class RelationNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, features_1, features_2):
        pair = torch.cat([features_1, features_2], dim=1)
        return self.layers(pair)


class DisjointRelationNet(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features_1, features_2):
        pair = torch.cat([features_1, features_2], dim=1)
        return self.layers(pair)


class Mapper(nn.Module):
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
