import torch
import os
from datasets import dataset
from datasets.classwise import ClasswiseSampler, ClasswisePairedSampler
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, ColorJitter, Normalize
from PIL import Image


def read_dataset(input_size, batch_size, root, set):
    if set in ['CUB', 'CAR', 'NABirds', 'Cotton', 'Soy']:
        path_train_data = os.path.join(root, 'train')
        path_test_data = os.path.join(root, 'test')
        print(f'Loading {set} trainset')
        # trainset = dataset.CUB(input_size=input_size, root=root, is_train=True)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        #                                           shuffle=True, num_workers=8, drop_last=False)
        # print('Loading CUB testset')
        # testset = dataset.CUB(input_size=input_size, root=root, is_train=False)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        #                                          shuffle=False, num_workers=8, drop_last=False)

        train_transform = Compose([Resize((input_size, input_size), Image.BILINEAR),
                                   RandomHorizontalFlip(),
                                   # ColorJitter(brightness=0.2, contrast=0.2),
                                   ToTensor(),
                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = Compose([Resize((input_size, input_size), Image.BILINEAR),
                                  ToTensor(),
                                  Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = torchvision.datasets.ImageFolder(root=path_train_data, transform=train_transform)
        # trainset = ClasswiseSampler(root=path_train_data, transform=train_transform)
        # trainset = ClasswisePairedSampler(root=path_train_data, transform=train_transform)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  pin_memory=True, num_workers=8, drop_last=False)
        print(f'Loading {set} testset')
        # testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testset = torchvision.datasets.ImageFolder(root=path_test_data, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                                 shuffle=False, num_workers=8, drop_last=False)

        trainevalset = torchvision.datasets.ImageFolder(root=path_train_data, transform=test_transform)
        trainevalloader = torch.utils.data.DataLoader(trainevalset, batch_size=batch_size, pin_memory=True,
                                                 shuffle=False, num_workers=8, drop_last=False)

    # elif set == 'CAR':
    #     print('Loading car trainset')
    #     trainset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=True)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                               shuffle=True, num_workers=8, drop_last=False)
    #     print('Loading car testset')
    #     testset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=False)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                              shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Aircraft':
        print('Loading Aircraft trainset')
        # trainset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=True)
        path_train_data = os.path.join(root, 'trainval')
        path_test_data = os.path.join(root, 'test')

        train_transform = Compose([Resize((input_size, input_size), Image.BILINEAR),
                                   RandomHorizontalFlip(),
                                   # ColorJitter(brightness=0.2, contrast=0.2),
                                   ToTensor(),
                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        test_transform = Compose([Resize((input_size, input_size), Image.BILINEAR),
                                   ToTensor(),
                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = torchvision.datasets.ImageFolder(root=path_train_data, transform=train_transform)
        # trainset = ClasswiseSampler(root=path_train_data, transform=train_transform)
        # trainset = ClasswisePairedSampler(root=path_train_data, transform=train_transform)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  pin_memory=True, num_workers=8, drop_last=False)
        print('Loading Aircraft testset')
        # testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testset = torchvision.datasets.ImageFolder(root=path_test_data, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                                 shuffle=False, num_workers=8, drop_last=False)

        trainevalset = torchvision.datasets.ImageFolder(root=path_train_data, transform=test_transform)
        trainevalloader = torch.utils.data.DataLoader(trainevalset, batch_size=batch_size, pin_memory=True,
                                                 shuffle=False, num_workers=8, drop_last=False)

    else:
        print('Please choose supported dataset')
        os._exit()

    return trainloader, testloader, trainevalloader