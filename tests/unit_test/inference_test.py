import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from inference.inference import predict

def test_predict():
    # 假設有一個測試模型和圖片
    print(os.getcwd())
    img_path = "data/hotdog/test/hotdog/1000.png"
    model_path = "checkpoints/model_epoch_1.pth"
    pred, prob = predict(img_path, model_path, device='cpu')
    assert pred in [0, 1]
    prob_arr = np.array(prob)
    assert ((prob_arr >= 0.0) & (prob_arr <= 1.0)).all()