import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path
from utils.AOLM import AOLM
import torchvision.transforms as T


class DisjointEncoder(nn.Module):
    def __init__(self, proposalN, num_classes, channels, device):
        super(DisjointEncoder, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.DEVICE = device

    def forward(self, x):
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        # raw branch
        raw_logits = self.rawcls_net(embedding)

        #SCDA
        coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(self.DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
            #local_imgs[i:i + 1] = T.Resize((448, 448))(T.CenterCrop(size=(256, 384))(x[i]))
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]

        five_crops = T.FiveCrop((224, 224))(local_imgs)  # [T.RandomResizedCrop(size=224)(local_imgs) for _ in range(5)]  # T.FiveCrop((224, 224))(local_imgs)
        five_crops = torch.cat(five_crops)
        _, crop_embeddings, _ = self.pretrained_model(five_crops)
        proposalN_windows_logits = self.rawcls_net(crop_embeddings)
        crop_embeddings = crop_embeddings.reshape(batch_size, 5, -1)

        proposalN_indices, proposalN_windows_scores, window_scores = None, None, None

        # proposalN_indices, proposalN_windows_scores, window_scores \
        #     = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs,\
               local_embeddings, crop_embeddings

