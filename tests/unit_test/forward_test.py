import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from model.models import HotDogsRecognizeModel

def test_model_forward():
    model = HotDogsRecognizeModel(num_classes=2, weights=None)
    x = torch.randn(2, 3, 224, 224)  # 假資料
    y = model(x)
    assert y.shape == (2, 2)