import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from models.relational_proxies import RelationalProxies
from models.global_only import GlobalOnly
from models.holistic_encoding import HolisticEncoding
from models.disjoint_encoding import DisjointEncoding
from models.transformer_agg import TransformerAgg
from models.gnn_agg_ondisk import GNNAggOnDisk
from models.gnn_agg_online import GNNAggOnline
from models.gnn_agg_hausdorff import GNNAggHausdorff
from models.multiview_hausdorff import MultiViewHausdorff
from networks.encoder import DisjointEncoder
from utils import constants
from utils.auto_load_resume import auto_load_resume


class Initializers:
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None

    def env(self):
        args = self.args
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
        else:
            print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
            self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            print("[INFO] Device type: " + str(self.device))

    def data(self):
        args = self.args
        print('[INFO] Dataset: {}'.format(args.dataset))
        # Stores (Number of classes, Number of local views) for each dataset.
        dataset_props = {'FGVCAircraft': (100, 7),
                         'StanfordCars': (196, 7),
                         'CUB': (200, 8),
                         'NABirds': (555, 8),
                         'iNaturalist': (5089, 8),
                         'CottonCultivar': (80, 7),
                         'SoyCultivar': (200, 7)}

        args.n_classes, args.n_local = dataset_props.get(args.dataset)
        if args.n_classes is None:
            print('[INFO] Dataset does not match. Exiting...')
            exit(1)

        if args.data_root is None:
            path_data = os.path.join(constants.HOME, 'Datasets', args.dataset)
        else:
            path_data = args.data_root

        # Note: for FGVCAircraft dataset, there are three splits.
        # We will use the trainval split to train the model.
        if args.dataset == 'FGVCAircraft':
            path_train_data = os.path.join(path_data, 'trainval')
        else:
            path_train_data = os.path.join(path_data, 'train')
        path_test_data = os.path.join(path_data, 'test')

        # Data generator
        print('[INFO] Setting data loader...', end='')

        train_transform = RProxyTransformTrain()
        test_transform = RProxyTransformTest()

        trainset = datasets.ImageFolder(root=path_train_data, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bsize, shuffle=True,
                                                  pin_memory=True, num_workers=args.num_workers, drop_last=False)
        testset = datasets.ImageFolder(root=path_test_data, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bsize, pin_memory=True,
                                                 shuffle=False, num_workers=args.num_workers, drop_last=False)
        print('Done', flush=True)

        return args, trainloader, testloader

    def modeltype(self):
        args, device = self.args, self.device
        # Get the pretrained backbone for extracting global-views
        backbone = DisjointEncoder(num_classes=args.n_classes, num_local=args.n_local, crop_mode=args.crop_mode)
        print("[INFO]", str(str(constants.BACKBONE)), "loaded in memory.")

        if args.logdir is None:
            logdir = os.path.join(args.checkpoint, args.dataset, 'logdir')
        else:
            logdir = args.logdir
        
        if args.model_type == 'relational_proxies':
            model = RelationalProxies(backbone, args.n_classes, logdir)
            print('[INFO] Model: Relational Proxies')
        elif args.model_type == 'global_only':
            model = GlobalOnly(backbone, args.n_classes, logdir, args.train_backbone)
            print('[INFO] Model: Global Only')
        elif args.model_type == 'holistic_encoding':
            model = HolisticEncoding(backbone, args.n_classes, logdir, args.train_backbone)
            print('[INFO] Model: Holistic Encoding')
        elif args.model_type == 'disjoint_encoding':
            model = DisjointEncoding(backbone, args.n_classes, logdir, args.train_backbone)
            print('[INFO] Model: Disjoint Encoding')
        elif args.model_type == 'transformer_agg':
            model = TransformerAgg(backbone, args.n_classes, logdir, args.train_backbone, args.local_weight)
            print('[INFO] Model: Transformer-based aggregation of local views')
        elif args.model_type == 'gnn_agg_ondisk':
            model = GNNAggOnDisk(backbone, args.n_classes, logdir, args.train_backbone, args.local_weight)
            print('[INFO] Model: GNN-based aggregation of local views (with on-disk staging)')
        elif args.model_type == 'gnn_agg_online':
            model = GNNAggOnline(backbone, args.n_classes, logdir, args.train_backbone, args.local_weight)
            print('[INFO] Model: GNN-based aggregation of local views (online)')
        elif args.model_type == 'gnn_agg_hausdorff':
            model = GNNAggHausdorff(backbone, args.n_classes, logdir, args.train_backbone, args.local_weight)
            print('[INFO] Model: GNN-based aggregation with Hausdorff distance')
        elif args.model_type == 'multiview_hausdorff':
            model = MultiViewHausdorff(backbone, args.n_classes, logdir, args.train_backbone, args.local_weight, args.recovery_epoch)
            print('[INFO] Model: GNN-based aggregation with Multi-view Hausdorff distance minimization')
        model.to(device)
        self.model = model

        return model

    def checkpoint(self):
        args, model = self.args, self.model
        save_path = os.path.join(args.checkpoint, args.dataset)
        if args.pretrained and os.path.exists(save_path):
            start_epoch, lr = auto_load_resume(model, save_path, status='train')
            assert start_epoch < constants.END_EPOCH
            model.lr = lr
            model.start_epoch = start_epoch
        else:
            os.makedirs(save_path, exist_ok=True)
            start_epoch = 0
        return save_path, start_epoch


class RProxyTransformTrain:
    def __init__(self):
        self.transfo = transforms.Compose([
            transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        return self.transfo(image)


class RProxyTransformTest:
    def __init__(self):
        self.trans = transforms.Compose([
            transforms.Resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        return self.trans(image)
