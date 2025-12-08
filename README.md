# HotDogRecognizer

HotDogRecognizer 是一個基於深度學習的影像分類專案，目標為辨識圖片中的熱狗與非熱狗。專案採用主流深度學習架構設計，支援訓練、推論、模型匯出、API 部署與自動化測試，方便一條龍開發與產品化。

---

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
│   └── api.py           # FastAPI 部署服務
│
├── logs/                # TensorBoard、wandb 等訓練日誌
│
├── checkpoints/         # 儲存模型權重
│
├── tests/               # 單元測試與整合測試
│   └── unit_test/
│   └── integration_test/
│
├── requirements.txt     # 套件需求
├── README.md            # 專案說明
└── main.py              # 主程式入口（整合 CLI）
```

---

## 功能模組說明

### 1. 訓練（train）
- 執行資料前處理、模型建立、訓練與驗證，產生最佳化模型權重。
- 支援 CLI 參數化設定（如 config 檔、device 選擇）。

### 2. 推論（infer）
- 載入訓練好的模型，對新圖片進行分類預測。
- 支援 CLI 單張推論。

### 3. 模型匯出（export）
- 將訓練好的模型轉換為 ONNX、TorchScript 等格式，方便跨平台部署。
- 支援 CLI 一鍵匯出。

### 4. API 部署（deploy）
- 以 FastAPI 部署推論服務，支援 HTTP API 圖片上傳與即時預測。
- 可直接用 CLI 啟動 API 服務，方便整合與測試。

### 5. 日誌與可視化
- 訓練過程自動記錄 loss、accuracy 等指標，支援 TensorBoard 監控。
- 提供訓練結果可視化工具。

### 6. 單元測試與整合測試
- `tests/unit_test/`：覆蓋模型、資料、推論等核心模組。
- `tests/integration_test/`：API 服務、CLI 流程等整合測試。
- 支援 pytest 自動化驗證。

---

## 快速開始

1. 安裝依賴套件
   ```bash
   pip install -r requirements.txt
   ```

2. 準備資料集
   - 將 hotdog 資料集放入 `data/hotdog/` 目錄。

3. 執行訓練
   ```bash
   python main.py train --config config.yaml
   ```

4. 單張圖片推論
   ```bash
   python main.py infer --img_path path/to/image.jpg --model_path checkpoints/model_epoch_10.pth
   ```

5. 匯出模型（ONNX/TorchScript）
   ```bash
   python main.py export --model_path checkpoints/model_epoch_10.pth --onnx_path export/hotdog.onnx --torchscript_path export/hotdog.pt
   ```

6. 啟動 API 服務
   ```bash
   python main.py deploy
   ```
   - 啟動後可用 HTTP API 上傳圖片進行即時預測。

7. 使用 TensorBoard 觀察訓練結果
   ```bash
   tensorboard --logdir=logs
   ```

8. 執行自動化測試
   ```bash
   pytest -s tests/unit_test/
   pytest -s tests/integration_test/
   ```

---

## 主要技術

- PyTorch
- torchvision
- FastAPI
- TensorBoard
- ONNX / TorchScript
- pytest
- d2l (Dive into Deep Learning)

---

## 開發規範與建議

- 所有模組皆有單元測試與整合測試，確保系統穩定。
- 參數集中管理於 config 檔案，方便調參與重現。
- 訓練、推論、匯出、部署皆可用 CLI 一鍵執行。
- 目錄結構清晰，易於團隊協作與維護。
- 歡迎 PR、issue 與討論，協助專案持續優化。

---

## 聯絡與貢獻

歡迎 issue、PR 或討論，協助專案持續優化！
