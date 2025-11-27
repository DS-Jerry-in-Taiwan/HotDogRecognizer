import torch
from torch import nn
import torchvision.models as models

class HotDogsRecognizeModel(nn.Module):
    """
    Use a pre-trained model to fine-tune for hotdog recognizer
    """
    def  __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        nn.init.xavier_uniform_(self.model.fc.weight)
        
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path, device='cpu', num_classes=2, pretrained=False):
        model = cls(num_classes=num_classes, pretrained=pretrained)
        model.load_state_dict(torch.load(path ,map_location=device))
        model = model.to(device)
        model.eval()
        return model
    
if __name__ == "__main__":
    model = HotDogsRecognizeModel(num_classes=2)
    print(model)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(y.shape)  # should be [4, 2]