#%%
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
    
    
normalize = torchvision.transforms.Normalize([0.485,0.456,0.406],
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

# Setting fine-tuning parameters
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) # 2 output classes
nn.init.xavier_uniform_(finetune_net.fc.weight)

# Tainning function
def train_finetune_net(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
    batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
    batch_size=batch_size, shuffle=False)
    
    device = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction = 'none')
    
    if param_group:
        params_1x = [param for name, param in net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, 
            weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter,  test_iter, loss ,trainer, num_epochs, device)
    
train_finetune_net(finetune_net,5e-5)

# %%
