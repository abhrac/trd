import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from networks.relation_net import DisjointRelationNet
from utils import constants
from functools import partial


class HolisticEncoding(nn.Module):
    def __init__(self, backbone, num_classes, logdir, train_backbone):
        super(HolisticEncoding, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR

        self.backbone = backbone
        self.aggregator = partial(torch.flatten, start_dim=-2, end_dim=-1)

        # self.relation_net = DisjointRelationNet(feature_dim=(self.feature_dim * (backbone.num_local + 1)),
        #                                 out_dim=self.feature_dim, num_classes=num_classes)
        self.relation_net = DisjointRelationNet(feature_dim=self.feature_dim * 2, out_dim=self.feature_dim, num_classes=num_classes)
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            trainable_params = chain(self.relation_net.parameters())
        else:
            trainable_params = chain(backbone.parameters(), self.relation_net.parameters())

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

            global_repr, summary_logits, relation_logits = self.compute_reprs(im)
            loss = self.criterion(relation_logits, labels) + self.criterion(summary_logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict(global_repr, summary_logits, relation_logits, labels, epoch_state)

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

                global_repr, summary_repr, relation_logits = self.compute_reprs(im)
                epoch_state = self.predict(global_repr, summary_repr, relation_logits, labels, epoch_state)

                loss = self.criterion(relation_logits, labels)
                epoch_state['loss'] += loss.item()

            self.post_epoch('Test', epoch, epoch_state, len(testloader.dataset), None)

    def compute_reprs(self, im):
        global_embed, local_embeds = self.backbone(im)
        _, summary_embed, _ = self.backbone.extractor(im)

        summary_repr = self.relation_net(summary_embed, summary_embed)
        relation_repr = self.relation_net(global_embed, global_embed)  # self.relation_net(global_embed, summary_repr)

        return global_embed, summary_repr, relation_repr

    @torch.no_grad()
    def predict(self, global_repr, summary_repr, relation_logits, labels, epoch_state):
        pred = relation_logits.max(1, keepdim=True)[1]
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
