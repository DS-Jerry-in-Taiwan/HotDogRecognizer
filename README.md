# HotDogRecognizer

HotDogRecognizer 是一個基於深度學習的影像分類專案，目標為辨識圖片中的熱狗與非熱狗。專案採用主流深度學習架構設計，方便訓練、分析與模型部署。

## 專案目錄結構

```
HotDogRecognizer/
│
├── data/                # 資料集與前處理腳本
│   └── hotdog/          # 原始資料
│
├── models/              # 模型定義與儲存
│   └── model.py
│
├── trainer/             # 訓練流程模組
│   └── train.py
│   └── trainer.py
│   └── data.py
│
├── utils/               # 工具函式（logger、可視化等）
│   └── logger.py
│   └── visualize.py
│   └── tensorboard_utils.py
│
├── inference/           # 推論與部署
│   └── inference.py
│   └── export.py
│
├── logs/                # TensorBoard、wandb 等訓練日誌
│
├── checkpoints/         # 儲存模型權重
│
├── requirements.txt     # 套件需求
├── README.md            # 專案說明
└── main.py              # 主程式入口（可選）
```

## 功能模組說明

### 1. 訓練流程模組
- 資料載入、資料增強、模型定義、訓練迴圈、優化器設定。
- 方便管理訓練邏輯與調參。

### 2. 訓練結果呈現
- 使用 TensorBoard、matplotlib 等工具即時或事後分析 loss、accuracy、混淆矩陣等。
- 方便模型調整與成效追蹤。

### 3. 模型結果輸出模組
- 模型儲存、載入、推論（inference）、部署。
- 方便後續應用、測試與部署。

## 快速開始

1. 安裝依賴套件
   ```bash
   pip install -r requirements.txt
   ```

2. 準備資料集
   - 將 hotdog 資料集放入 `data/hotdog/` 目錄。

3. 執行訓練
   ```bash
   python trainer/train.py
   ```

4. 使用 TensorBoard 觀察訓練結果
   ```bash
   tensorboard --logdir=logs
   ```

5. 推論與模型部署
   - 參考 `inference/inference.py` 進行模型推論。

## 主要技術

- PyTorch
- torchvision
- TensorBoard
- d2l (Dive into Deep Learning)

## 聯絡與貢獻

歡迎 issue、PR 或討論，協助專案持續優化！
