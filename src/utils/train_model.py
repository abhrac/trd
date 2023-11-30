import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, device
from utils.eval_model import eval
import random
import torch.nn.functional as F
from losses.moco import MoCo
from pytorch_metric_learning.losses import ProxyAnchorLoss


def train(model,
          aggregator,
          relation_net,
          trainloader,
          testloader,
          trainevalloader,
          criterion,
          contrast_intra_1,
          contrast_intra_2,
          contrast_inter,
          sinkhorn_div,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):
    kl_div = torch.nn.KLDivLoss(reduction='sum')
    contrast_rel = MoCo().to(device)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            # if set == 'CUB':
            #     im1, labels, _, _ = data
            # else:
            im1, im2, labels = data
            im1, im2, labels = im1.to(device), im2.to(device), labels.to(device)

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _, \
            local_embed_1, crop_embeds_1 = model(im1, epoch, i, 'train')

            local_embed_2, crop_embeds_2 = model(im2, epoch, i, 'train')[-2:]

            # similarities = torch.bmm(crop_embeddings_1, crop_embeddings_2.permute(0, 2, 1))
            # distributions = torch.nn.functional.softmax(similarities, dim=-1)
            # kld_loss = -torch.nn.functional.kl_div(distributions.unsqueeze(dim=-1), distributions.unsqueeze(dim=-2), reduction='sum')
            # kld_loss.backward()

            # if epoch > 100:
            #     prob_div = torch.tensor(0)
            #     for inst_distributions in distributions:
            #         for (idx1, d1) in enumerate(inst_distributions):
            #             indices = torch.Tensor(range(5)) != idx1
            #             prob_div = prob_div + sinkhorn_div(d1.unsqueeze(dim=0), inst_distributions[indices])
            #             # for (idx2, d2) in enumerate(inst_distributions):
            #             #     # prob_div = prob_div - torch.nn.functional.kl_div(d1, d2, reduction='sum')
            #             #     prob_div = sinkhorn_div(d1, d2)

            moco_crop_idx = random.randint(0, proposalN - 1)
            moco_crop_embed_1 = crop_embeds_1[:, moco_crop_idx, :]
            moco_crop_embed_2 = crop_embeds_2[:, moco_crop_idx, :]

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            # contrastive_intra_loss = criterion(*contrast_intra_1(moco_crop_embed_1, local_embed_1)) + \
            #                          criterion(*contrast_intra_2(moco_crop_embed_2, local_embed_2))
            # contrastive_inter_loss = criterion(*contrast_inter(local_embed_1, local_embed_2))
            # contrastive_loss = contrastive_intra_loss + contrastive_inter_loss

            # logits, labels = contrastive_global(crop_embeddings_2, local_embeddings)
            # contrastive_loss = contrastive_loss + criterion(logits, labels)
            # logits, labels = contrastive_local(crop_embeddings_1, crop_embeddings_2)
            # contrastive_loss = contrastive_loss + criterion(logits, labels)
            # F_i, G_j = sinkhorn_div(crop_embeddings_2, crop_embeddings_1)

            total_loss = raw_loss + local_loss + windowscls_loss  # + contrastive_loss
            if 50000 < epoch < 100000:
                contrastive_intra_loss = criterion(*contrast_intra_1(moco_crop_embed_1, local_embed_1)) + \
                                         criterion(*contrast_intra_2(moco_crop_embed_2, local_embed_2))
                contrastive_inter_loss = criterion(*contrast_inter(local_embed_1, local_embed_2))
                contrastive_loss = contrastive_intra_loss + contrastive_inter_loss
                total_loss = total_loss + contrastive_loss
            if epoch > 100000:
                # contrastive_intra_loss = criterion(*contrast_intra_1(moco_crop_embed_1, local_embed_1)) + \
                #                          criterion(*contrast_intra_2(moco_crop_embed_2, local_embed_2))
                # contrastive_inter_loss = criterion(*contrast_inter(local_embed_1, local_embed_2))
                # contrastive_loss = contrastive_intra_loss + contrastive_inter_loss
                # total_loss = total_loss + contrastive_loss
                #
                #
                attr_repr_1 = aggregator(crop_embeds_1)
                attr_repr_2 = aggregator(crop_embeds_2)
                #
                # r1 = torch.sqrt(((local_embed_1 - attr_repr_1) ** 2).sum(dim=1))
                # r2 = torch.sqrt(((local_embed_2 - attr_repr_2) ** 2).sum(dim=1))
                # relational_loss = F.smooth_l1_loss(r1, r2)

                r1 = relation_net(local_embed_1, attr_repr_1)
                # r2 = relation_net(local_embed_2, attr_repr_2)
                # relational_loss = criterion(*contrast_rel(r1, r2))

                relational_loss = criterion(model.rawcls_net(r1), labels)

                total_loss = total_loss + relational_loss
            # if epoch > 200:
            #     softmax = torch.nn.Softmax(dim=-1)
            #     # prob_div = sinkhorn_div(softmax(crop_embeddings_1), softmax(crop_embeddings_2)).sum()
            #     prob_div = 0.5 * (kl_div(softmax(crop_embeddings_1).log(), softmax(crop_embeddings_2))
            #                       + kl_div(softmax(crop_embeddings_2).log(), softmax(crop_embeddings_1)))
            #     total_loss = total_loss + prob_div  # + contrastive_loss  # + sinkhorn_loss

            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # evaluation every epoch
        if eval_trainset and epoch % 2000 == 0:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg \
                = eval(model, trainevalloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:
                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        if epoch % 2000 == 0:
            # eval testset
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
            local_loss_avg \
                = eval(model, testloader, criterion, 'test', save_path, epoch)

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
            if (epoch % save_interval == 0) or (epoch == end_epoch):
                print('Saving checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

            # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
            # and delete the redundant ones
            checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
            if len(checkpoint_list) == max_checkpoint_num + 1:
                idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
                min_idx = min(idx_list)
                os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))


def train_cls(model,
          aggregator,
          relation_net,
          trainloader,
          testloader,
          trainevalloader,
          criterion,
          contrast_intra_1,
          contrast_intra_2,
          contrast_inter,
          sinkhorn_div,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):
    relational_proxy_anchor_criterion = ProxyAnchorLoss(num_classes=200, embedding_size=2048).to(device)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            # if set == 'CUB':
            #     im1, labels, _, _ = data
            # else:
            im, labels = data
            im, labels = im.to(device), labels.to(device)

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _, \
            local_embed_1, crop_embeds_1 = model(im, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            total_loss = raw_loss + local_loss
            if epoch > 100:
                total_loss = total_loss + windowscls_loss

                attr_repr_1 = aggregator(crop_embeds_1)
                r1 = relation_net(local_embed_1, attr_repr_1)
                relational_proxy_anchor_criterion.proxies = model.rawcls_net.weight
                relational_loss = relational_proxy_anchor_criterion(r1, labels)
                # relational_loss = criterion(model.rawcls_net(r1), labels)

                total_loss = total_loss + relational_loss

            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # evaluation every epoch
        if eval_trainset and epoch % 1 == 0:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg \
                = eval(model, trainevalloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:
                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        if epoch % 1 == 0:
            # eval testset
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
            local_loss_avg \
                = eval(model, testloader, criterion, 'test', save_path, epoch)

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
            if (epoch % save_interval == 0) or (epoch == end_epoch):
                print('Saving checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

            # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
            # and delete the redundant ones
            checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
            if len(checkpoint_list) == max_checkpoint_num + 1:
                idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
                min_idx = min(idx_list)
                os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))
