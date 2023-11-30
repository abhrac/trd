#!/usr/bin/env python

# system, numpy
import os
import random
import numpy as np
from utils.graph_gen import graph_gen
from utils.eval_model import eval

# pytorch, torch vision
import torch
from torchvision import datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.encoder import DisjointEncoder

# user defined
import misc
from options import Options
import sys

import config
from config import input_size, home, end_epoch, batch_size, proposalN, channels
from utils.auto_load_resume import auto_load_resume


def main(args):
    # Parse options
    args = Options().parse()

    # Header of the method
    if args.id != "":
        header = str(args.method) + "_" + str(args.id) + "_" + str(args.dataset) + "_" + str(
            args.backbone) + "_seed_" + str(args.seed)
    else:
        header = str(args.method) + "_" + str(args.dataset) + "_" + str(args.backbone) + "_seed_" + str(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Manual seed
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")

    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available.")

    # Information printing
    # tot_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("[INFO]", str(str(args.backbone)), "loaded in memory.")
    # print("[INFO] Embedding size:", str(backbone.embed_dim))
    # print("[INFO] Feature extractor TOT trainable params: " + str(tot_params))
    print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
    device = torch.device(config.device) # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type: " + str(device))

    # Dataset
    print('[INFO] Dataset: {}'.format(args.dataset))

    if args.dataset == 'CUB':
        args.n_classes = 200
    elif args.dataset == 'FGVCAircraft':
        args.n_classes = 100
    elif args.dataset == 'StanfordCars':
        args.n_classes = 196
    elif args.dataset == 'NABirds':
        args.n_classes = 555
    elif args.dataset == 'CottonCultivar':
        args.n_classes = 80
    elif args.dataset == 'SoyCultivar':
        args.n_classes = 200
    elif args.dataset == 'iNat':
        args.n_classes = 5089
    else:
        args.dataset = 'FGVCAircraft'
        print('[INFO] Dataset does not match. Exiting...')
        exit(1)

    # Get the pretrained backbone and adjust its patch embedding layer and heads
    backbone = DisjointEncoder(proposalN=proposalN, num_classes=args.n_classes, channels=channels, device=device)

    # Paths for dataset and checkpoint
    path_data = os.path.join(home, 'Datasets', args.dataset)
    # Note: for FGVCAircraft dataset, there are three splits.
    # We will use the trainval split to train the model.
    if args.dataset == 'FGVCAircraft':
        path_train_data = os.path.join(path_data, 'trainval')
    else:
        path_train_data = os.path.join(path_data, 'train')
    path_test_data = os.path.join(path_data, 'test')

    # Data generator
    print('[INFO] Setting data loader...', end='')

    train_transform = misc.RProxyTransformTrain(input_size)
    test_transform = misc.RProxyTransformTest(input_size)

    trainset = datasets.ImageFolder(root=path_train_data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=4, drop_last=False)
    testset = datasets.ImageFolder(root=path_test_data, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                              shuffle=False, num_workers=4, drop_last=False)

    args.length_train_loader = len(train_loader)
    print('Done', flush=True)

    # Define the model
    if args.method == 'rproxy':
        from models.rproxy import RelationalProxy
        model = RelationalProxy(backbone, args.n_classes)
        print('[INFO] Model: Relational Proxy')
    elif args.method == 'ablation_disjoint_only':
        from models.ablations.disjoint_only import DisjointOnly
        model = DisjointOnly(backbone, args.n_classes)
        print('[INFO] Model Ablation: Disjoint Only')
    elif args.method == 'ablation_no_vit':
        from models.ablations.no_vit import NoViT
        model = NoViT(backbone, args.n_classes)
        print('[INFO] Model Ablation: No ViT')
    elif args.method == 'ablation_no_rnet':
        from models.ablations.no_rnet import NoRNet
        model = NoRNet(backbone, args.n_classes)
        print('[INFO] Model Ablation: No RNet')
    elif args.method == 'naive':
        from models.naiveproxies import Naive
        model = Naive(backbone, args.n_classes)
        print('[INFO] Model: Naive')
    elif args.method == 'graphrel':
        from models.graphrel_hausdorff import GraphRel
        model = GraphRel(backbone, args.n_classes)
        print('[INFO] Model: GraphRel')
    else:  # method == 'graphrel':
        from models.graphrel_hausdorff_ex import GraphRel
        model = GraphRel(backbone, args.n_classes)
        print('[INFO] Model: GraphRelEx')
    # Transfer the model to device
    model.to(device)

    # NOTE: the checkpoint must be loaded AFTER
    # the model has been allocated into the device.

    save_path = os.path.join('./checkpoint', args.dataset)
    if os.path.exists(save_path) and args.pretrained:
        # start_epoch, lr = auto_load_resume(backbone, save_path, status='train')
        start_epoch, lr = auto_load_resume(model, save_path, status='train', device=device)
        assert start_epoch < end_epoch
        model.lr = lr
        model.start_epoch = start_epoch
    else:
        os.makedirs(save_path, exist_ok=True)
        start_epoch = 0

    for epoch in range(start_epoch + 1, end_epoch + 1):
        print('Training %d epoch' % epoch)
        # model.train_one_epoch(train_loader, test_loader, epoch=epoch, eval_fn=graph_gen, save_path=save_path)
        model.train_one_epoch(train_loader, test_loader, epoch=epoch, eval_fn=eval, save_path=save_path)


if __name__ == "__main__":
    main(sys.argv)
