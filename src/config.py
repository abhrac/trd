from os.path import expanduser

CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training
home = expanduser('~')
device = 'cuda:0'
model_name = ''

batch_size = 8
vis_num = batch_size  # The number of visualized images in tensorboard
eval_trainset = False  # Whether or not evaluate trainset
save_interval = 50
max_checkpoint_num = 200
end_epoch = 1000  # 200
init_lr = 0.001
lr_milestones = [60, 100]  # , 9841, 9850, 9860, 9900, 9950, 10000]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 448
num_crops = 5

# The pth path of localizer model
pretrain_path = './src/localizer/resnet50-19c8e357.pth'
proposalN = 5
