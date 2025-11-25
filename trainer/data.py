import os 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Data loading function
def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Build and return training and testing data loaders.
    Parameters:
    - data_dir (str): Path to the dataset directory.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of subprocesses to use for data loading.
    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - test_loader (DataLoader): DataLoader for the testing dataset.
    """

    # main flow:
    # 1. Define data augmentations functions
    # 2. define datasets using ImageFolder
    # 3. define data loaders
    
    # Define data augmentations functions
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # define datasets using ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_transforms
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_transforms
    )
    
    # define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader