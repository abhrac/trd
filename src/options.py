#!/usr/bin/env python

import misc
import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training script for the unsupervised phase via self-supervision")

        # Model parameters
        parser.add_argument("--backbone", default="ResNet50", help="Backbone: vit_base_patch16_448")
        parser.add_argument("--method", default="rproxy", help="Backbone: rproxy")
        parser.add_argument("--out_dim", default=4096, type=int,
                            help="""Dimensionality of the DINO head output. For complex and large datasets large values 
                            (like 65k) work well.""")
        parser.add_argument('--norm_last_layer', default=True, type=misc.bool_flag,
                            help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing
                                    leads to better performance but can make the training unstable. In our experiments,
                                    we typically set this paramater to False with vit_small and True with vit_base.""")
        parser.add_argument('--momentum_teacher', default=0.996, type=float,
                            help="""Base EMA parameter for teacher update. The value is increased to 1 during training 
                                    with cosine schedule. We recommend setting a higher value with small batches: for 
                                    example use 0.9995 with batch size of 256.""")
        parser.add_argument('--use_bn_in_head', default=False, type=misc.bool_flag,
                            help="Whether to use batch normalizations in projection head (Default: False)")

        # Temperature teacher parameters
        parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                            help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                            Try decreasing it if the training loss does not decrease.""")
        parser.add_argument('--teacher_temp', default=0.04, type=float,
                            help="""Final value (after linear warmup) of the teacher temperature. For most experiments,
                            anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and 
                            increase this slightly if needed.""")
        parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                            help='Number of warmup epochs for the teacher temperature (Default: 30).')

        # Training/Optimization parameters
        parser.add_argument('--use_fp16', type=misc.bool_flag, default=True,
                            help="""Whether or not to use half precision for training. Improves training time and memory 
                            requirements, but can provoke instability and slight decay of performance. We recommend 
                            disabling mixed precision if the loss is unstable, if reducing the patch size or if training 
                            with bigger ViTs.""")
        parser.add_argument('--weight_decay', type=float, default=0.04,
                            help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of 
                            training works well.""")
        parser.add_argument('--weight_decay_end', type=float, default=0.4,
                            help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger 
                            decay by the end of training improves performance for ViTs.""")
        parser.add_argument('--clip_grad', type=float, default=3.0,
                            help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm 
                            .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
        parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                            help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
        parser.add_argument("--batch_size", default=64, type=int, help="Size of the mini-batch")
        parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
        parser.add_argument('--freeze_last_layer', default=1, type=int,
                            help="""Number of epochs during which we keep the output layer fixed. Typically doing so 
                            during the first epoch helps training. Try increasing this value if the loss does not 
                            decrease.""")
        parser.add_argument("--lr", default=0.0005, type=float,
                            help="""Learning rate at the end of linear warmup (highest LR used during training). The
                            learning rate is linearly scaled with the batch size, and specified here for a reference 
                            batch size of 256.""")
        parser.add_argument("--warmup_epochs", default=10, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument('--min_lr', type=float, default=1e-6,
                            help="""Target LR at the end of optimization. We use a cosine LR schedule with linear 
                            warmup.""")
        parser.add_argument('--optimizer', default='adamw', type=str,
                            choices=['adamw', 'sgd', 'lars'],
                            help="""Type of optimizer. We recommend using adamw with ViTs.""")
        parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

        # Multi-crop parameters
        # parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.1, 1.),
                            help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                            Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                            recommend using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
        # parser.add_argument('--local_crops_number', type=int, default=8,
        parser.add_argument('--local_crops_number', type=int, default=0,
                            help="""Number of small local views to generate. Set this parameter to 0 to disable 
                            multi-crop training. When disabling multi-crop we recommend to use
                            "--global_crops_scale 0.14 1." """)
        parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                            help="""Scale range of the cropped image before resizing, relatively to the origin image.
                            Used for small local view cropping of multi-crop.""")

        # Misc
        parser.add_argument("--seed", default=-1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
        parser.add_argument("--epoch_start", default=0, type=int,
                            help="Epoch to start learning from, used when resuming")
        parser.add_argument("--dataset", default="CUB", help="Dataset: FGVCAircraft")
        parser.add_argument("--pretrained", action='store_true', default=False,
                            help="Whether to load pretrained weights")
        parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
        parser.add_argument("--checkpoint", default="", help="Location of a checkpoint file, used to resume training")
        parser.add_argument("--num_workers", default=8, type=int,
                            help="Number of torchvision workers used to load data (default: 8)")
        parser.add_argument("--image-size", default=448, type=int, help="Image size")
        parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")

        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
