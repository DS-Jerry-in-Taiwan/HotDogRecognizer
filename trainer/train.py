import os
import torch
from config import TrainConfig
torch.backends.cudnn.benchmark = True  # 加速卷積運算
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from trainer.data import get_data_loaders
from model.models import HotDogsRecognizeModel

# Initialize configuration
cfg = TrainConfig()

# training parameters
data_dir = cfg.data_dir
batch_size = cfg.batch_size
num_workers = cfg.num_workers
num_epochs = cfg.num_epochs
learning_rate = cfg.learning_rate
device = cfg.device
print(f'Using device: {device}')

# get data loaders
train_loader, test_loader = get_data_loaders(data_dir, batch_size, num_workers)

# initialize model
model = HotDogsRecognizeModel(num_classes=2, pretrained=True)
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.001)
writer = SummaryWriter(log_dir='logs') # TensorBoard writer

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss, train_acc, n = 0.0, 0.0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * y.size(0)
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.size(0)
        
    avg_loss = running_loss / n
    avg_acc = train_acc / n
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_acc, epoch)
    with torch.no_grad():
        model.eval()
        test_loss, test_acc, n_test = 0.0, 0.0, 0
        X,y = next(iter(test_loader))
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        test_loss += criterion(y_hat, y).item()
        test_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n_test += y.size(0)
        
    avg_test_loss = test_loss / n_test
    avg_test_acc = test_acc / n_test
    writer.add_scalar("Loss/test", avg_test_loss, epoch)
    writer.add_scalar("Accuracy/test", avg_test_acc, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
    
    # save model checkpoint
    checkpoint_path = os.path.join('checkpoints', f'model_epoch_{epoch+1}.pth')
    os.makedirs('checkpionts', exist_ok=True)
    
    torch.save(model.state_dict(), checkpoint_path)
writer.close()
