import torch.nn as nn
import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, device
from pytorch_metric_learning.losses import ProxyAnchorLoss
from itertools import chain

from networks.gcn import GCN
from networks.relation_net import RelationNet
from networks.vit import ViTf
from torch.optim.lr_scheduler import MultiStepLR
import config
from config import num_crops
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from typing import Union, List, Tuple
from torch_geometric.data import DataLoader


# class LocalGraphs(InMemoryDataset):
#     def __init__(self, root=''):
#         super(LocalGraphs, self).__init__(root=root)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def processed_file_names(self) -> Union[str, List[str], Tuple]:
#         return ['./local_graphs.dataset']
#
#     def download(self):
#         pass
#
#     @property
#     def raw_file_names(self) -> Union[str, List[str], Tuple]:
#         return []
#
#     def process(self):
#         node_lists, edge_lists = torch.load('graphs/node_lists.pth', map_location=device), torch.load('graphs/edge_lists.pth', map_location=device)
#         data_list = []
#         for i, node_list in enumerate(node_lists):
#             data_list.append(Data(x=node_list, edge_index=edge_lists[i]))
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


class Naive(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Naive, self).__init__()
        self.num_classes = num_classes
        self.lr = config.init_lr

        self.backbone = backbone
        self.aggregator = ViTf(num_inputs=1+num_crops, dim=2048, depth=3, heads=3, mlp_dim=256).to(device)
        # self.aggregator = nn.Linear(2048, 2048)
        self.relation_net = RelationNet(feature_dim=2048).to(device)

        # self.optimizer = torch.optim.SGD(backbone.parameters(), lr=self.lr, momentum=0.9, weight_decay=config.weight_decay)

        self.optimizer = torch.optim.SGD(chain(
            backbone.parameters(), self.aggregator.parameters(), self.relation_net.parameters()),
            lr=self.lr, momentum=0.9, weight_decay=config.weight_decay)

        self.scheduler = MultiStepLR(self.optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)
        self.criterion = nn.CrossEntropyLoss()

    # @staticmethod
    # def graphgen(node_embeddings):
    #     sims = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
    #     sims = sims * torch.ones(num_crops, num_crops).fill_diagonal_(0).cuda(device)  # disregard self-similarities
    #     directed: Tensor = sims > (sims.sum(dim=2) / num_crops-1).unsqueeze(dim=2)  # average only over non-zero elms
    #     undirected = directed + directed.transpose(1, 2)
    #     assert torch.all(undirected == undirected.transpose(1, 2)).item()  # validate symmetrization
    #     edges = undirected.nonzero()
    #
    #     edge_lists = []
    #     offset = 0
    #     for i, sample in enumerate(undirected):
    #         num_edges = undirected[i].sum()
    #         edges_i = edges[offset: offset + num_edges]
    #         # Edge list in COO format
    #         edges_i = edges_i[:, 1:].T
    #         edge_lists.append(edges_i)
    #         offset = offset + num_edges
    #
    #     torch.save(node_embeddings, f'graphs/node_lists.pth')
    #     torch.save(edge_lists, f'graphs/edge_lists.pth')

    def train_one_epoch(self, trainloader, testloader, epoch, eval_fn, save_path):
        self.backbone.train()
        epoch_loss = 0
        raw_correct = 0
        crops_correct = 0
        rel_correct = 0
        for i, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            self.optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _, \
            local_embed, crop_embeds = self.backbone(im)

            # crop_logits = self.backbone.rawcls_net(crop_embeds).reshape((-1, self.num_classes))
            # labels_crops = labels.unsqueeze(1).repeat((1, num_crops)).flatten()

            all_view_embeds = torch.cat([local_embed.unsqueeze(dim=1), crop_embeds], dim=1)
            attr_repr = self.aggregator(all_view_embeds)
            # attr_repr = self.aggregator(local_embed.unsqueeze(dim=1).repeat((1, 2, 1)))
            # attr_repr = self.aggregator(local_embed)
            attr_logits = self.backbone.rawcls_net(attr_repr)

            relation_repr = self.relation_net(local_embed, attr_repr)
            relation_logits = self.backbone.rawcls_net(relation_repr)

            loss = self.criterion(relation_logits, labels) + self.criterion(attr_logits, labels) + self.criterion(local_logits, labels) + self.criterion(raw_logits, labels)
            # loss = self.criterion(crop_logits, labels_crops).sum() + self.criterion(local_logits, labels).sum() + self.criterion(raw_logits, labels).sum()
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            pred_local = local_logits.max(1, keepdim=True)[1]
            raw_correct += pred_local.eq(labels.view_as(pred_local)).sum().item()

            pred_attr = attr_logits.max(1, keepdim=True)[1]
            crops_correct += pred_attr.eq(labels.view_as(pred_attr)).sum().item()

            pred_rel = relation_logits.max(1, keepdim=True)[1]
            rel_correct += pred_rel.eq(labels.view_as(pred_rel)).sum().item()

        self.scheduler.step()

        raw_accuracy = raw_correct / len(trainloader.dataset)
        crops_accuracy = crops_correct / len(trainloader.dataset)
        rel_accuracy = rel_correct / len(trainloader.dataset)

        print(f'Train loss: {epoch_loss}')
        print(f'Train Local Accuracy: {raw_accuracy * 100}%')
        print(f'Train Crops Accuracy: {crops_accuracy * 100}%')
        print(f'Train Relation Accuracy: {rel_accuracy * 100}%')

        # test_accuracy = eval_fn(self.backbone, testloader, self.criterion)
        # print(f'Test Accuracy: {test_accuracy * 100}%')

        if (epoch % config.save_interval == 0) or (epoch == config.end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'learning_rate': self.lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
