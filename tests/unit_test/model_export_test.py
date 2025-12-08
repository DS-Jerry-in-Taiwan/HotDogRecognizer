import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import os
from model.models import HotDogsRecognizeModel

def test_save_and_load_checkpoint(tmp_path):
    model = HotDogsRecognizeModel(num_classes=2, weights=None)
    save_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), save_path)
    model2 = HotDogsRecognizeModel(num_classes=2, weights=None)
    model2.load_state_dict(torch.load(save_path))
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)