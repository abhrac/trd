from typing import Union, List, Tuple

import torch
from tqdm import tqdm
from config import proposalN, device
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from networks.gcn import GCN
import torch.nn.functional as F


class LocalGraphs(InMemoryDataset):
    def __init__(self, root=''):
        super(LocalGraphs, self).__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['./local_graphs.dataset']

    def download(self):
        pass

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def process(self):
        node_lists, edge_lists = torch.load('node_lists.pth', map_location=device), torch.load('edge_lists.pth', map_location=device)
        data_list = []
        for i, node_list in enumerate(node_lists):
            data_list.append(Data(x=node_list, edge_index=edge_lists[i]))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_edges(attn_wts):
    attn_wts = attn_wts.mean(dim=1)  # Aggregate output from all heads
    attn_wts = attn_wts[:, 1:, 1:]
    # Add an edge if the attention weight is greater than the self-attention value (diagonal elements)
    adj = attn_wts > attn_wts.diagonal(dim1=1, dim2=2).unsqueeze(dim=1)
    undir = adj + adj.transpose(dim0=1, dim1=2)
    edges = undir.nonzero()
    offset = 0
    edge_lists = []
    for i, sample in enumerate(undir):
        num_edges = undir[i].sum()
        edges_i = edges[offset: offset + num_edges]
        # Edge list in COO format
        edges_i = edges_i[:, 1:].T
        edge_lists.append(edges_i)
    return edge_lists


def graph_gen(disj_enc, testloader, criterion, aggregator, relation_net):
    disj_enc.eval()
    aggregator.eval()
    relation_net.eval()
    print('Evaluating')
    num_nodes = 5
    node_embed_criterion = torch.nn.CosineEmbeddingLoss()

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    raw_correct = 0
    local_correct = 0

    # edge_lists = []
    # node_lists = []
    # with torch.no_grad():
    #     for i, data in enumerate(tqdm(testloader)):
    #         images, labels = data
    #         images = images.to(device)
    #         labels = labels.to(device)
    #
    #         proposalN_windows_score, proposalN_windows_logits, indices, \
    #         window_scores, coordinates, raw_logits, local_logits, local_imgs, local_embed, crop_embeds = model(images)
    #
    #         attr_repr = aggregator(crop_embeds)
    #         attn_wts = aggregator.get_attn_wts()
    #         edge_lists.extend(get_edges(attn_wts))
    #         node_lists.extend(crop_embeds)
    #     torch.save(node_lists, 'node_lists.pth')
    #     torch.save(edge_lists, 'edge_lists.pth')

    rel_dict, lab_dict = {}, {}
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, coordinates, raw_logits, local_logits, local_imgs, local_embed, crop_embeds = disj_enc(images)
            attr_repr = aggregator(crop_embeds)
            r = relation_net(local_embed, attr_repr)
            rel_dict[i] = r
            lab_dict[i] = labels

    dataset = LocalGraphs()
    dataloader = DataLoader(dataset, batch_size=8)
    gcn = GCN(num_in_features=2048, num_out_features=2048).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-3)

    for epoch in range(100):
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(device)
            relations, labels = rel_dict[batch_idx], lab_dict[batch_idx]
            out = gcn(data)
            out, proxies = F.normalize(out), F.normalize(disj_enc.rawcls_net.weight)
            sims = torch.mm(out, proxies.T).reshape(data.num_graphs, num_nodes, len(proxies))
            idx0 = torch.arange(data.num_graphs).unsqueeze(dim=-1).repeat(1, num_nodes).flatten()
            idx1 = torch.arange(num_nodes).unsqueeze(dim=0).repeat(1, data.num_graphs).flatten()
            idx_labs = labels.unsqueeze(dim=1).repeat(1, num_nodes).flatten()
            # sims[:, :, 0].argsort(dim=1, descending=True)
            sims = sims[idx0, idx1, idx_labs].reshape(data.num_graphs, num_nodes)  # Similarity w/ the ground-truth proxy
            relevances = (sims > sims.mean(dim=1).unsqueeze(dim=1))  # +ve/-ve-s wrt the class-proxy
            # positives = out.reshape(8, 5, -1)[relevances.unsqueeze(dim=-1).repeat(1, 1, 2048)]
            # negatives = out.reshape(8, 5, -1)[relevances.logical_not().unsqueeze(dim=-1).repeat(1, 1, 2048)]
            positives = out.reshape(data.num_graphs, num_nodes, -1)[relevances]
            negatives = out.reshape(data.num_graphs, num_nodes, -1)[relevances.logical_not()]
            instwise_pos = relevances.sum(dim=-1)
            instwise_neg = relevances.logical_not().sum(dim=-1)
            pos_tgt, neg_tgt = [], []
            for i in range(data.num_graphs):
                pos_tgt.append(relations[i].repeat(instwise_pos[i], 1))
            pos_tgt = torch.cat(pos_tgt)
            for i in range(data.num_graphs):
                neg_tgt.append(relations[i].repeat(instwise_neg[i], 1))
            neg_tgt = torch.cat(neg_tgt)

            nodes_partitioned = torch.cat([positives, negatives])
            rels_partitioned = torch.cat([pos_tgt, neg_tgt])
            relv_labs = torch.cat([torch.ones(len(positives)), -torch.ones(len(negatives))]).to(device)

            loss = node_embed_criterion(nodes_partitioned, rels_partitioned, relv_labs)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Node embedding loss: {loss.item()}')
            breakpoint()



    #
    #         r = relation_net(local_embed, attr_repr)
    #
    #         raw_loss = criterion(raw_logits, labels)
    #         local_loss = criterion(local_logits, labels)
    #         windowscls_loss = criterion(proposalN_windows_logits,
    #                                     labels.unsqueeze(1).repeat(1, proposalN).view(-1))
    #
    #         total_loss = raw_loss + local_loss + windowscls_loss
    #
    #         raw_loss_sum += raw_loss.item()
    #         local_loss_sum += local_loss.item()
    #         windowscls_loss_sum += windowscls_loss.item()
    #
    #         total_loss_sum += total_loss.item()
    #
    #         # raw
    #         pred = raw_logits.max(1, keepdim=True)[1]
    #         raw_correct += pred.eq(labels.view_as(pred)).sum().item()
    #         # local
    #         local_logits = (model.rawcls_net(r) + model.rawcls_net(attr_repr) + local_logits) / 3
    #         # local_logits = (model.rawcls_net(r) + local_logits) / 2
    #         pred = local_logits.max(1, keepdim=True)[1]
    #         local_correct += pred.eq(labels.view_as(pred)).sum().item()
    #
    #
    # raw_loss_avg = raw_loss_sum / (i + 1)
    # local_loss_avg = local_loss_sum / (i + 1)
    # windowscls_loss_avg = windowscls_loss_sum / (i + 1)
    # total_loss_avg = total_loss_sum / (i + 1)
    #
    # raw_accuracy = raw_correct / len(testloader.dataset)
    # local_accuracy = local_correct / len(testloader.dataset)
    #
    # return raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
    #        local_loss_avg
