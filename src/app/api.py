from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 
from src.ml.predict import Predictor

app = FastAPI(title="M3 Model API")

predictor = Predictor("params.yaml")

class PredictRequest(BaseModel):
    x:list[float] = Field(
        ..., 
        description="Flat list of 3072 floats representing a 3x32x32 RGB image (C*H*W).",
        examples=[[0.0] * 3072],
    )

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predictor.predict(req.x)
        return {
            "class_id": result["class_id"],
            "label": result["label"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))