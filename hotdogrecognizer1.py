#%%
# Cell 1: 匯入套件
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#%%
# Cell 2: 設定資料路徑
data_dir = "./data/hotdog"

# Cell 3: 載入資料集
train_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# Cell 4: 顯示部分 hotdog/not-hotdog 圖片
hotdogs = [train_images[i][0] for i in range(8)]
not_hotdogs = [train_images[-i -1][0] for i in range(8)]
for img in hotdogs + not_hotdogs:
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Cell 5: 定義資料增強
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

# Cell 6: 初始化預訓練模型
device = d2l.try_all_gpus()
pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net = pretrained_net.to(device[0])
# Cell 7: 設定微調模型
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) # 2 output classes
nn.init.xavier_uniform_(finetune_net.fc.weight)
finetune_net = finetune_net.to(device[0])  # 移到 GPU（放在最後）

# Cell 8: 定義訓練函式（加入 TensorBoard 支援）
def train_finetune_net(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size, shuffle=False)
    
    # device = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    # net = net.to(device[0])
    writer = SummaryWriter(log_dir='logs')  # TensorBoard writer

    if param_group:
        params_1x = [param for name, param in net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, 
            weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device[0]), y.to(device[0])
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            trainer.step()
            train_loss += float(l.sum())
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        avg_loss = train_loss / n
        avg_acc = train_acc / n
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_acc, epoch)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

    writer.close()
#%%
# Cell 9: 執行訓練

train_finetune_net(finetune_net, 5e-5)
# %%
