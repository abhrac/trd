import torch.nn as nn
import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, device
from pytorch_metric_learning.losses import ProxyAnchorLoss
from itertools import chain
from networks.relation_net import RelationNet
from networks.vit import ViTf
from torch.optim.lr_scheduler import MultiStepLR
import config


class RelationalProxy(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RelationalProxy, self).__init__()
        self.lr = config.init_lr
        self.backbone = backbone

        self.aggregator = ViTf(num_inputs=5, dim=2048, depth=3, heads=3, mlp_dim=256).to(device)
        self.relation_net = RelationNet(feature_dim=2048).to(device)

        self.optimizer = torch.optim.SGD(chain(
            backbone.parameters(), self.aggregator.parameters(), self.relation_net.parameters()),
                                    lr=self.lr, momentum=0.9, weight_decay=config.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.relational_proxy_anchor_criterion = ProxyAnchorLoss(num_classes=num_classes, embedding_size=2048).to(device)

    def train_one_epoch(self, trainloader, testloader, epoch, eval_fn, save_path):
        self.backbone.train()
        lr = next(iter(self.optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            self.optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _, \
            local_embed, crop_embeds = self.backbone(im)

            raw_loss = self.criterion(raw_logits, labels)
            local_loss = self.criterion(local_logits, labels)
            windowscls_loss = self.criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            # total_loss = raw_loss + local_loss
            if epoch > 0:
                # total_loss = total_loss + windowscls_loss

                attr_repr = self.aggregator(crop_embeds)
                attr_loss = 0
                if epoch > 25:
                    attr_loss = self.relational_proxy_anchor_criterion(attr_repr, labels)

                r = self.relation_net(local_embed, attr_repr)
                self.relational_proxy_anchor_criterion.proxies = self.backbone.rawcls_net.weight
                relational_loss = self.relational_proxy_anchor_criterion(r, labels)
                # relational_loss = criterion(model.rawcls_net(r1), labels)

                # total_loss = total_loss + attr_loss + relational_loss
                total_loss = relational_loss

            total_loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        #
        # # evaluation every epoch
        # if eval_trainset and epoch % 1 == 0:
        #     raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg \
        #         = eval_fn(self.backbone, trainloader, self.criterion, self.aggregator, self.relation_net)
        #
        #     print(
        #         'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
        #             100. * raw_accuracy, 100. * local_accuracy))
        #
        #     # tensorboard
        #     with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:
        #         writer.add_scalar('Train/learning rate', lr, epoch)
        #         writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
        #         writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
        #         writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
        #         writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
        #         writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
        #         writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        if epoch % 1 == 0:
            # eval testset
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
            local_loss_avg \
                = eval_fn(self.backbone, testloader, self.criterion, self.aggregator, self.relation_net)

            print(
                'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
                writer.add_scalar('Test/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Test/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Test/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Test/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

            # save checkpoint
            # if (epoch % config.save_interval == 0) or (epoch == config.end_epoch):
            #     print('Saving checkpoint')
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.backbone.state_dict(),
            #         'learning_rate': lr,
            #     }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
        # if (epoch % config.save_interval == 0) or (epoch == config.end_epoch):
        #     print('Saving checkpoint')
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': self.state_dict(),
        #         'learning_rate': lr,
        #     }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

