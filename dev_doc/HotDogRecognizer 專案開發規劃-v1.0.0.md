# HotDogRecognizer 專案開發規劃（主流框架設計建議）

## 開發階段規劃

### 第一階段：資料準備與前處理
**目標：**
- 完成資料集收集、整理與前處理，確保訓練資料品質。


1. **資料集收集與分類**
   - 收集熱狗與非熱狗圖片，分別存放於 `data/hotdog/train/hotdog`、`data/hotdog/train/not-hotdog`、`data/hotdog/test/hotdog`、`data/hotdog/test/not-hotdog` 目錄。
   - 確認每類資料數量充足且標註正確。

2. **資料集目錄結構規劃與建立**
   - 建立資料夾結構：
     ```
     data/
       hotdog/
         train/
           hotdog/
           not-hotdog/
         test/
           hotdog/
           not-hotdog/
     ```
   - 檢查資料夾權限與路徑正確。

3. **資料前處理腳本開發**
   - 撰寫 `trainer/data.py`，實現圖片 resize、normalize、augmentation（如隨機裁切、翻轉等）。
   - 測試前處理腳本，確保能正確載入並處理所有圖片。

4. **資料載入模組開發**
   - 在 `trainer/data.py` 中實作 PyTorch 的 `ImageFolder` 與 `DataLoader`，支援 batch 載入與 shuffle。
   - 加入資料增強流程，並可設定 train/test 不同的 transform。
   - 測試資料載入模組，確保能正確回傳 batch 資料。

5. **驗證資料前處理與載入流程**
   - 隨機抽取部分圖片，顯示前處理後的結果（可用 matplotlib）。
   - 檢查資料分布、類別比例，確保無誤。


**Checklist：**
- [x] 資料集收集與分類（hotdog / not-hotdog）
- [x] 資料集目錄結構規劃與建立
- [x] 資料前處理腳本（resize、normalize、augmentation）
- [x] 資料載入模組（`trainer/data.py`）

---

### 第二階段：模型設計與訓練流程
**目標：**
- 建立模型結構、訓練迴圈與優化器設定，完成基本訓練流程。

1. **模型定義**
   - 在 `models/model.py` 建立模型結構（可用 torchvision 預訓練模型或自訂 CNN）。
   - 設定輸出類別數（2類：hotdog / not-hotdog）。
   - 測試模型 forward，確保能正確輸出。

2. **訓練流程模組開發**
   - 在 `trainer/train.py` 或 `trainer/trainer.py` 撰寫訓練主程式與訓練迴圈。
   - 整合資料載入、模型、優化器、損失函數。
   - 支援訓練參數（batch size、learning rate 等）設定。

3. **優化器與損失函數設定**
   - 選擇主流優化器（如 SGD、Adam）。
   - 使用交叉熵損失函數（`nn.CrossEntropyLoss`）。

4. **訓練參數管理**
   - 支援參數化設定（可用 argparse、config 檔案或 yaml）。
   - 設定 epoch、batch size、learning rate 等。

5. **支援 GPU 訓練**
   - 檢查 CUDA 是否可用，將模型與資料移至 GPU。
   - 測試 GPU 訓練流程，確保無 device mismatch 錯誤。

6. **訓練流程測試**
   - 執行訓練主程式，確認 loss、accuracy 能正常收斂。
   - 儲存訓練日誌與模型 checkpoint，方便後續分析與部署。

**Checklist：**
- [x] 模型定義（`models/model.py`）
- [x] 訓練流程模組（`trainer/train.py`、`trainer/trainer.py`）
- [x] 優化器與損失函數設定
- [x] 訓練參數（batch size、learning rate 等）管理
- [x] 支援 GPU 訓練

---

### 第三階段：訓練結果紀錄與分析
**目標：**
- 整合 TensorBoard、matplotlib 等工具，實現訓練過程的即時監控與分析。


1. **TensorBoard 日誌紀錄模組開發**
   - 在 `utils/tensorboard_utils.py` 封裝 TensorBoard 日誌紀錄功能。
   - 訓練主程式中呼叫，記錄 loss、accuracy 等指標。

2. **訓練指標可視化**
   - 使用 TensorBoard 或 matplotlib 畫出 loss、accuracy 隨 epoch 變化曲線。
   - 撰寫簡單的 `utils/visualize.py`，可讀取日誌或訓練結果並繪圖。

3. **訓練結果分析腳本開發**
   - 在 `utils/visualize.py` 實作訓練過程與結果分析（如 loss/accuracy 曲線、混淆矩陣）。
   - 可加入錯誤分析（如預測錯誤的圖片展示）。

4. **混淆矩陣與錯誤分析**
   - 訓練結束後，利用 sklearn 或 matplotlib 計算並繪製混淆矩陣。
   - 分析模型在不同類別上的表現，找出易混淆的類別。

5. **測試與驗證**
   - 執行訓練主程式，確認 TensorBoard 日誌能正確紀錄並可視化。
   - 執行分析腳本，驗證 loss/accuracy 曲線與混淆矩陣能正確產生。

**Checklist：**
- [x] TensorBoard 日誌紀錄（`utils/tensorboard_utils.py`）
- [x] 訓練指標（loss、accuracy）可視化
- [x] 訓練結果分析腳本（`utils/visualize.py`）
- [x] 混淆矩陣、錯誤分析

---

### 第四階段：模型儲存、推論與部署
**目標：**
- 完成模型儲存、載入、推論與部署流程，支援後續應用。

### 第四階段：模型儲存、推論與部署 — 開發步驟

1. **模型儲存與載入**
   - 在訓練結束後，將模型權重存到 `checkpoints/` 目錄（`torch.save`）。
   - 在 `models/model.py` 或新檔案實作模型載入函式（`torch.load` + `model.load_state_dict`）。

2. **推論模組開發**
   - 建立 `inference/inference.py`，撰寫模型推論流程（載入模型、資料前處理、預測、回傳結果）。
   - 支援單張圖片或批次推論。

3. **模型匯出（ONNX、TorchScript）**
   - 在 `inference/export.py` 實作模型匯出功能，支援 ONNX 或 TorchScript 格式，方便部署到不同平台。
   - 測試匯出後模型能正確推論。

4. **部署腳本開發**
   - 根據需求，撰寫 API（如 FastAPI）、Web（如 Streamlit）、Batch（批次推論）等部署腳本。
   - 整合推論模組，支援外部呼叫與結果回傳。

5. **測試與驗證**
   - 測試模型儲存/載入流程，確保模型能正確還原。
   - 測試推論模組，確認輸入/輸出正確。
   - 測試匯出模型能在目標平台正確運作。
   - 測試 API/Web/Batch 部署腳本，確保可用性與穩定性。

**Checklist：**
- [ ] 模型儲存與載入（`checkpoints/`、`models/model.py`）
- [ ] 推論模組（`inference/inference.py`）
- [ ] 模型匯出（ONNX、TorchScript，`inference/export.py`）
- [ ] 部署腳本（API、Web、Batch）

---

### 第五階段：系統整合與優化
**目標：**
- 完成各模組整合，優化系統效能與可維護性。

**Checklist：**
- [ ] 主程式入口（`main.py`）
- [ ] 參數設定與配置管理（config 檔案）
- [ ] 單元測試與整合測試
- [ ] 專案文件與使用說明（`README.md`）
- [ ] 套件需求（`requirements.txt`）

---

## 補充建議
- 每階段完成後，進行 code review 與測試。
- 可逐步導入 wandb、MLflow 等進階監控工具。
- 保持目錄結構清晰，方便團隊協作與維護。

---
**依照以上規劃逐步開發，可確保專案架構符合主流深度學習框架設計思路。**