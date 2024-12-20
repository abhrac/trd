import os
from itertools import chain

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from networks.relation_net import DisjointRelationNet, Mapper
from networks.class_proxy import ProxyHead, create_proxy_graphs
from networks.gcn import GCN
from utils import constants


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


def cdist(set1, set2):
    ''' Pairwise Distance between two matrices
    Input:  x is a Nxd matrix
            y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # dist = set1.unsqueeze(1) - set2.unsqueeze(0)
    dist = set1.unsqueeze(1) - set2.unsqueeze(2)
    return dist.abs()


def instance_wise_hausdorff_distance(g1, g2):
    num_nodes = g1.shape[1]
    g1_local, g2_local = g1[:, 1:], g2[:, 1:]
    g1_global, g2_global = g1[:, 0], g2[:, 0]

    dist_matrix = cdist(g1_local, g2_local).pow(2.).sum(-1)

    d1 = 0.5 + g1_global.abs()
    d2 = 0.5 + g2_global.abs()

    pw_cost_1, _ = dist_matrix.min(1)
    pw_cost_2, _ = dist_matrix.min(2)

    cost_g1 = torch.min(torch.cat([d1, pw_cost_2], dim=-1), dim=-1)[0] / (num_nodes * 2)
    cost_g2 = torch.min(torch.cat([d2, pw_cost_1], dim=-1), dim=-1)[0] / (num_nodes * 2)

    return cost_g1, cost_g2

def hausdorff_distance(g1, g2):
    cost_g1, cost_g2 = instance_wise_hausdorff_distance(g1, g2)
    return cost_g1.sum() + cost_g2.sum()


class ProxyGraph(nn.Module):
    def __init__(self, backbone, num_classes, logdir, train_backbone, local_weight, recovery_epoch):
        super(ProxyGraph, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR
        self.local_weight = local_weight
        self.recovery_epoch = recovery_epoch

        self.backbone = backbone
        self.aggregator = GCN(num_in_features=self.feature_dim, num_out_features=self.feature_dim)
        self.num_local = backbone.num_local
        self.proxy_heads = [ProxyHead(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes) for _ in range(self.num_local)]

        self.relation_net = DisjointRelationNet(feature_dim=self.feature_dim * 2, out_dim=self.feature_dim, num_classes=num_classes)
        self.global_net = Mapper(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes)
        self.agg_net = Mapper(feature_dim=self.feature_dim, out_dim=self.feature_dim, num_classes=num_classes)

        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            trainable_params = chain(self.relation_net.parameters(), self.global_net.parameters(), self.agg_net.parameters())
        else:
            trainable_params = chain(backbone.parameters(), self.relation_net.parameters(), self.global_net.parameters(),
                                     self.agg_net.parameters(), self.aggregator.parameters())

        self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)
        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(logdir)

    def train_one_epoch(self, trainloader, epoch, save_path):
        print('Training %d epoch' % epoch)
        self.train()
        device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
        recovery_weight = (epoch == self.recovery_epoch) * self.local_weight
        epoch_state = {'loss': 0, 'correct': 0}
        self.proxy_heads = [head.to(device) for head in self.proxy_heads]
        for _, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            self.optimizer.zero_grad()

            global_logits, local_logits, sem_logits, semrel_graphs = self.compute_reprs(im)
            view_loss = self.criterion(sem_logits + (recovery_weight * local_logits) + global_logits, labels)

            assignment_logits = torch.cat(
                    [head(semrel_graphs.flatten(0,1)).reshape(
                        im.shape[0], self.num_local, -1).max(dim=1)[0].unsqueeze(dim=0)
                        for head in self.proxy_heads]).max(dim=0)[0]
            assignments = torch.cat([head(semrel_graphs.flatten(0,1)) for head in self.proxy_heads]) # Duplication, should ideally be resolved.
            # Useful for fetching specific proxy-view assignment scores and computing a mask matrix.
            # sample_idx = torch.tensor(range(len(im))).unsqueeze(-1).expand(-1, self.num_local).flatten().unsqueeze(-1).to(device)
            # view_idx = torch.tensor([range(self.num_local)]).expand(len(im), -1).flatten().unsqueeze(-1).to(device)

            view_labels = labels.unsqueeze(-1).expand(-1, self.num_local).flatten()
            view_labels = view_labels.unsqueeze(-1).expand(-1, len(self.proxy_heads)).flatten()
            assignment_loss = self.criterion(assignments, view_labels) * (epoch == self.recovery_epoch)

            proxy_graphs, proxy_graph_embeds = create_proxy_graphs(self.proxy_heads)
            minibatch_proxy_graph_labels = [proxy_graph_embeds[i] for i in labels]
            minibatch_proxy_embed_labels = proxy_graph_embeds[labels]
            hdist = hausdorff_distance(semrel_graphs, minibatch_proxy_embed_labels) * (epoch == self.recovery_epoch)

            loss = view_loss + recovery_weight * (assignment_loss + hdist)

            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict(global_logits, local_logits, sem_logits, assignment_logits, labels, epoch_state)

        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            print('Testing %d epoch' % epoch)
            self.eval()
            device = self.relation_net.layers[0].weight.device  # hacky, but keeps the arg list clean
            self.proxy_heads = [head.to(device) for head in self.proxy_heads]
            epoch_state = {'loss': 0, 'correct': 0}
            for i, data in enumerate(tqdm(testloader)):
                im, labels = data
                im, labels = im.to(device), labels.to(device)

                global_repr, local_repr, relation_logits, semrel_graphs = self.compute_reprs(im)
                assignment_logits = torch.cat(
                    [head(semrel_graphs.flatten(0,1)).reshape(
                        im.shape[0], self.num_local, -1).max(dim=1)[0].unsqueeze(dim=0)
                        for head in self.proxy_heads]).max(dim=0)[0]
                assignments = torch.cat([head(semrel_graphs.flatten(0,1)) for head in self.proxy_heads]) # Duplication, should ideally be resolved.

                view_labels = labels.unsqueeze(-1).expand(-1, self.num_local).flatten()
                view_labels = view_labels.unsqueeze(-1).expand(-1, len(self.proxy_heads)).flatten()
                assignment_loss = self.criterion(assignments, view_labels) * (epoch == self.recovery_epoch)

                proxy_graphs, proxy_graph_embeds = create_proxy_graphs(self.proxy_heads)
                minibatch_proxy_graph_labels = [proxy_graph_embeds[i] for i in labels]
                minibatch_proxy_embed_labels = proxy_graph_embeds[labels]
                hdist = instance_wise_hausdorff_distance(semrel_graphs, minibatch_proxy_embed_labels)

                epoch_state = self.predict(global_repr, local_repr, relation_logits, assignment_logits, labels, epoch_state)

                loss = self.criterion(relation_logits, labels)
                epoch_state['loss'] += loss.item()

            self.post_epoch('Test', epoch, epoch_state, len(testloader.dataset), None)

    def compute_reprs(self, im):
        global_embed, local_embeds, global_view, local_views = self.backbone.forward_with_views(im)
        _, full_embed, _ = self.backbone.extractor(im)
        
        graphs = graphgen(local_embeds)
        graph_loader = DataLoader(graphs, batch_size=len(im))
        semrel_graphs = self.aggregator(next(iter(graph_loader)))
        semrel_graphs = semrel_graphs.reshape(len(im), local_embeds.shape[1], -1)  # batch_size x num_local
        agg_embed = semrel_graphs.mean(dim=1)

        local_logits = self.agg_net(agg_embed) + self.global_net(full_embed)
        global_logits = self.global_net(global_embed) + self.global_net(full_embed)
        sem_logits = self.relation_net(global_embed, agg_embed) + self.global_net(full_embed)

        return global_logits, local_logits, sem_logits, semrel_graphs

    def compute_views(self, im, num_views=2):
        views = []
        for _ in range(num_views):
            global_embed, local_embeds, global_view, local_views = self.backbone.forward_with_views(im)
            _, full_embed, _ = self.backbone.extractor(im)
            
            graphs = graphgen(local_embeds)
            graph_loader = DataLoader(graphs, batch_size=len(im))
            semrel_graphs = self.aggregator(next(iter(graph_loader)))
            view = semrel_graphs.reshape(len(im), local_embeds.shape[1], -1)  # batch_size x num_local
            views.append(view)

        return views

    @torch.no_grad()
    def predict(self, global_logits, local_logits, relation_logits, assignment_logits, labels, epoch_state):

        pred = (global_logits + (self.local_weight * local_logits) + relation_logits + assignment_logits).max(1, keepdim=True)[1]
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
