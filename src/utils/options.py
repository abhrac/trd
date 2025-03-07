import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training script for Relational Proxies")

        # Data
        parser.add_argument("--data_root", default=None, help="Root directory for all datasets. Default: ~/Datasets/")
        parser.add_argument("--dataset", default="FGVCAircraft",
                            help="FGVCAircraft, StanfordCars, CUB, NABirds, iNaturalist, CottonCultivar, SoyCultivar")
        parser.add_argument("--train-bsize", default=8, type=int, help="Train batch size.")
        parser.add_argument("--test-bsize", default=128, type=int, help="Test batch size.")

        # Model
        parser.add_argument("--model-type", default="relational_proxies",
                            help="Choice of algorithm.")
        parser.add_argument("--checkpoint", default="./checkpoint/",
                            help="Location of a checkpoint file, used to resume training.")
        parser.add_argument("--logdir", default=None,
                            help="Location of logging directory. Default: ./checkpoint/logdir/")
        parser.add_argument("--pretrained", action='store_true', default=False,
                            help="Whether to load pretrained weights")
        parser.add_argument("--train-backbone", action='store_true', default=False,
                            help="Whether to update backbone weights.")
        parser.add_argument("--crop-mode", default="random",
                            help="Process of local view extraction. Options: random, five_crop")
        parser.add_argument("--local-weight", default=1e-4, type=float, help="Weight of local views.")
        parser.add_argument("--recovery-epoch", default=1, type=int, help="Beginning of the transitivity recovery phase.")
        parser.add_argument("--task", default="fgvc", help="Type of task. Choose from ['fgvc']")
        parser.add_argument("--backbone-type", default="disjoint_encoder",
                            help="Type of backbone for the task. Note that backbones are task-specific. Choose from ['disjoint_encoder']")

        # Misc
        parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
        parser.add_argument("--num_workers", default=4, type=int,
                            help="Number of torchvision workers used to load data (default: 4)")
        parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
        parser.add_argument("--eval_only", action='store_true', default=False,
                            help="No training. Evaluate only on pretrained weights.")

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
