import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from trainer.data import get_data_loaders

def test_data_loader():
    train_loader, test_loader = get_data_loaders('data/hotdog', batch_size=2, num_workers=0)
    x, y = next(iter(train_loader))
    assert x.shape[0] == 2
    assert y.shape[0] == 2