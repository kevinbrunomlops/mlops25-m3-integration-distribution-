from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ml.predict import Predictor

app = FastAPI(title="M3 Model API")

predictor = Predictor("params.yaml")

class PredictRequest(BaseModel):
    x:list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predictor.predict(req.x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))