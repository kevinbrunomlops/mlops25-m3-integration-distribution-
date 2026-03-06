from fastapi import FastAPI, HTTPException, UploadFile, File
from src.ml.predict import Predictor

import io 
import numpy as np 
from PIL import Image

app = FastAPI(title="M3 Model API")

predictor = Predictor("params.yaml")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((32, 32))

        arr = np.array(img).astype("float32") / 255.0
        arr = arr.transpose(2, 0, 1) #HWC -> CHW
        x = arr.reshape(-1).tolist() #3x32x32 = 3072

        return predictor.predict(x)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    