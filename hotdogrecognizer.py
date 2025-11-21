import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# d2l.DATA_DIR = os.path.abspath('./data')
# os.makedirs('./data', exist_ok=True)

# d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffaadea4d3f2f6f3f6bf25f1a8d47f5f6e6')
# data_dir = d2l.download_extract('hotdog')
# print(data_dir)

data_dir = "./data/hotdog"
# load train and test datasets
train_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_images[i][0] for i in range(8)]
not_hotdogs = [train_images[-i -1][0] for i in range(8)]
# for img in hotdogs + not_hotdogs:
#     plt.imshow(img)
#     plt.axis('off')
    # plt.show()
    
    
normalize = tochvision.tranforms.Normalize([0.485,0.456,0.406],
                                           [0.229,0.224,0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# Define and Initialize pre-trained model
pretrained_net = torchvision.models.resnet18(pretrained=True)
