from fastapi import FastAPI, UploadFile, File
from inference.inference import predict
import shutil
import os

app = FastAPI()

MODEL_PATH = "checkpoints/model_epoch_100.pth"

# 啟動方式（專案根目錄下）：
# uvicorn inference.api:app --reload

@app.post("/predict/")
async def predict_hotdog(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    pred, prob = predict(temp_file_path, MODEL_PATH, device='cpu')
    os.remove(temp_file_path) # Clean up the temporary file
    return {"filename": file.filename, "prediction": int(pred), "probabilities": prob.tolist()}

