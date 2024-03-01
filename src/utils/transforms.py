import torchvision.transforms as transforms

from PIL import Image
from utils import constants


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
