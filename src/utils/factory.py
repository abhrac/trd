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


class Factory:
    def __init__(self, args) -> None:
        self.args = args
        self.backbones = Backbones(args)
        self.executors = Executors(args)

    def get_backbone(self, task):
        return self.backbones.get(task)
    
    def get_executor(self, model_type, backbone):
        return self.executors.get(model_type, backbone)


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
