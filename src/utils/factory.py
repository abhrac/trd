import os
import torch
from torchvision import datasets

from networks.encoder import DisjointEncoder

from models.relational_proxies import RelationalProxies
from models.global_only import GlobalOnly
from models.holistic_encoding import HolisticEncoding
from models.disjoint_encoding import DisjointEncoding
from models.transformer_agg import TransformerAgg
from models.gnn_agg_ondisk import GNNAggOnDisk
from models.gnn_agg_online import GNNAggOnline
from models.gnn_agg_hausdorff import GNNAggHausdorff
from models.multiview_hausdorff import MultiViewHausdorff

from utils import constants
from utils.transforms import RProxyTransformTrain, RProxyTransformTest


class Factory:
    def __init__(self, args) -> None:
        self.args = args
        self.backbones = Backbones(args)
        self.executors = Executors(args)
        self.data = Data(args)

    def get_backbone(self, task):
        return self.backbones.get(task)
    
    def get_executor(self, model_type, backbone):
        return self.executors.get(model_type, backbone)
    
    def get_data(self, task):
        return self.data.get(task)


class Backbones:
    def __init__(self, args) -> None:
        self.args = args
        self.backbones = {'fgvc': self._disjoint_encoder}

    def _disjoint_encoder(self):
        args = self.args
        return DisjointEncoder(num_classes=args.n_classes, num_local=args.n_local, crop_mode=args.crop_mode)
    
    def get(self, task):
        return self.backbones[task]()


class Executors:
    def __init__(self, args) -> None:
        self.args = args
        self.executors = {'relational_proxies': self._relational_proxies,
                          'global_only':self._global_only,
                          'holistic_encoding':self._holistic_encoding,
                          'disjoint_encoding':self._disjoint_encoding,
                          'transformer_agg':self._transformer_agg,
                          'gnn_agg_ondisk':self._gnn_agg_on_disk,
                          'gnn_agg_online':self._gnn_agg_online,
                          'gnn_agg_hausdorff':self._gnn_agg_hausdorff,
                          'multiview_hausdorff':self._multiview_hausdorff}

    def _relational_proxies(self, backbone):
        args = self.args
        print('[INFO] Model: Relational Proxies')
        return RelationalProxies(backbone, args.n_classes, args.logdir)

    def _global_only(self, backbone):
        args = self.args
        print('[INFO] Model: Global Only')
        return GlobalOnly(backbone, args.n_classes, args.logdir, args.train_backbone)
    
    def _holistic_encoding(self, backbone):
        args = self.args
        print('[INFO] Model: Holistic Encoding')
        return HolisticEncoding(backbone, args.n_classes, args.logdir, args.train_backbone)

    def _disjoint_encoding(self, backbone):
        args = self.args
        print('[INFO] Model: Disjoint Encoding')
        return DisjointEncoding(backbone, args.n_classes, args.logdir, args.train_backbone)

    def _transformer_agg(self, backbone):
        args = self.args
        print('[INFO] Model: Transformer-based aggregation of local views')
        return TransformerAgg(backbone, args.n_classes, args.logdir, args.train_backbone, args.local_weight)

    def _gnn_agg_on_disk(self, backbone):
        args = self.args
        print('[INFO] Model: GNN-based aggregation of local views (with on-disk staging)')
        return GNNAggOnDisk(backbone, args.n_classes, args.logdir, args.train_backbone, args.local_weight)

    def _gnn_agg_online(self, backbone):
        args = self.args
        print('[INFO] Model: GNN-based aggregation of local views (online)')
        return GNNAggOnline(backbone, args.n_classes, args.logdir, args.train_backbone, args.local_weight)

    def _gnn_agg_hausdorff(self, backbone):
        args = self.args
        print('[INFO] Model: GNN-based aggregation with Hausdorff distance')
        return GNNAggHausdorff(backbone, args.n_classes, args.logdir, args.train_backbone, args.local_weight)

    def _multiview_hausdorff(self, backbone):
        args = self.args
        print('[INFO] Model: GNN-based aggregation with Multi-view Hausdorff distance minimization')
        return MultiViewHausdorff(backbone, args.n_classes, args.logdir, args.train_backbone, args.local_weight, args.recovery_epoch)
    
    def get(self, model_type, backbone):
        return self.executors[model_type](backbone)


class Data:
    def __init__(self, args) -> None:
        self.args = args
        self.data_managers = {'fgvc': self.fgvc}

    def fgvc(self):
        args = self.args
        print('[INFO] Dataset: {}'.format(args.dataset))
        # Stores (Number of classes, Number of local views) for each dataset.
        self.dataset_props = {'FGVCAircraft': (100, 7),
                         'StanfordCars': (196, 7),
                         'CUB': (200, 8),
                         'NABirds': (555, 8),
                         'iNaturalist': (5089, 8),
                         'CottonCultivar': (80, 7),
                         'SoyCultivar': (200, 7)}
        args.n_classes, args.n_local = self.dataset_props.get(args.dataset)
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
    
    def get(self, task):
        return self.data_managers[task]()

