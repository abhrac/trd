import os
import shutil
import glob
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data

from networks.relation_net import DisjointRelationNet
from networks.gcn import GCN
from utils import constants

from typing import Union, List, Tuple


def graphgen(node_embeddings):
    device = node_embeddings.device
    _, num_nodes, _ = node_embeddings.shape
    sims = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
    sims = sims * torch.ones(num_nodes, num_nodes).fill_diagonal_(0).to(device)  # disregard self-similarities
    directed: Tensor = sims > (sims.sum(dim=2) / num_nodes - 1).unsqueeze(dim=2)  # average only over non-zero elms
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

class GNNAggOnline(nn.Module):
    def __init__(self, backbone, num_classes, logdir, train_backbone, local_weight):
        super(GNNAggOnline, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR
        self.local_weight = local_weight

        self.backbone = backbone
        self.aggregator = GCN(num_in_features=self.feature_dim, num_out_features=self.feature_dim)

        self.relation_net = DisjointRelationNet(feature_dim=self.feature_dim * 2, out_dim=self.feature_dim, num_classes=num_classes)
        self.agg_net = DisjointRelationNet(feature_dim=self.feature_dim * 2, out_dim=self.feature_dim, num_classes=num_classes)

        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            trainable_params = chain(self.relation_net.parameters(), self.agg_net.parameters())
        else:
            trainable_params = chain(backbone.parameters(), self.relation_net.parameters(),
                                     self.agg_net.parameters(), self.aggregator.parameters())

        self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)
        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(logdir)

    def train_one_epoch(self, trainloader, epoch, save_path):
        print('Training %d epoch' % epoch)
        self.train()
        device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
        epoch_state = {'loss': 0, 'correct': 0}
        for i, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            self.optimizer.zero_grad()

            global_logits, local_logits, sem_logits = self.compute_reprs(im)
            loss = self.criterion(sem_logits + (self.local_weight * local_logits) + global_logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict(global_logits, local_logits, sem_logits, labels, epoch_state)

        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            print('Testing %d epoch' % epoch)
            self.eval()
            device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
            epoch_state = {'loss': 0, 'correct': 0}
            for i, data in enumerate(tqdm(testloader)):
                im, labels = data
                im, labels = im.to(device), labels.to(device)

                global_repr, local_repr, relation_logits = self.compute_reprs(im)
                epoch_state = self.predict(global_repr, local_repr, relation_logits, labels, epoch_state)

                loss = self.criterion(relation_logits, labels)
                epoch_state['loss'] += loss.item()

            self.post_epoch('Test', epoch, epoch_state, len(testloader.dataset), None)

    def compute_reprs(self, im):
        global_embed, local_embeds, global_view, local_views = self.backbone.forward_with_views(im)
        _, full_embed, _ = self.backbone.extractor(im)
        
        graphs = graphgen(local_embeds)
        graph_loader = DataLoader(graphs, batch_size=len(im))
        semrel_graphs = self.aggregator(next(iter(graph_loader)))
        gcn_embed = semrel_graphs.reshape(len(im), local_embeds.shape[1], -1)  # batch_size x num_local
        agg_embed = gcn_embed.mean(dim=1)

        local_logits = self.agg_net(agg_embed, agg_embed)
        global_logits = self.relation_net(global_embed, global_embed)
        sem_logits = self.relation_net(global_embed, agg_embed)

        return global_logits, local_logits, sem_logits

    @torch.no_grad()
    def predict(self, global_logits, local_logits, relation_logits, labels, epoch_state):
        pred = (global_logits + (self.local_weight * local_logits) + relation_logits).max(1, keepdim=True)[1]
        epoch_state['correct'] += pred.eq(labels.view_as(pred)).sum().item()

        return epoch_state

    @torch.no_grad()
    def post_epoch(self, phase, epoch, epoch_state, num_samples, save_path):
        accuracy = epoch_state['correct'] / num_samples
        loss = epoch_state['loss']

        print(f'{phase} Loss: {loss}')
        print(f'{phase} Accuracy: {accuracy * 100}%')
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)

        if (phase == 'Train') and ((epoch % constants.SAVE_EVERY == 0) or (epoch == constants.END_EPOCH)):
            self.scheduler.step()
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'learning_rate': self.lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

    def post_job(self):
        """Post-job actions"""
        self.writer.flush()
        self.writer.close()
