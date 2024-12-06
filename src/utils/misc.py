import os
import random

import numpy as np
import torch

from utils import constants
from utils.auto_load_resume import auto_load_resume
from utils.factory import Factory


class Initializers:
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.task = args.task
        self.backbone_type = args.backbone_type

        if args.logdir is None:
            args.logdir = os.path.join(args.checkpoint, args.dataset, 'logdir')

        self.factory = Factory(args)

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
        return self.factory.get_data(self.args.task)

    def modeltype(self):
        args, device = self.args, self.device
        # Get the pretrained backbone for extracting views
        backbone = self.factory.get_backbone(self.task, self.backbone_type)
        print("[INFO]", str(str(constants.BACKBONE)), "loaded in memory.")

        model = self.factory.get_executor(args.model_type, backbone)
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
