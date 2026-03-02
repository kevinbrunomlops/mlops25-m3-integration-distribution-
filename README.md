# MLOPS25 - M3: Integration & Distribution
## Overview

This project demonstrates a professional MLOps workflow where a trained PyTorch model is:
1. Exported to TorchScript
2. Integrated into a FastApi application
3. Containerized using docker 
4. Developed colloboratively using feature branches and pull request.

The models performs image classification on 32x32 RGB images (CIFAR-10 format)

## Model Description
The models is a Convolutional Neural Network (CNN) orginally trained in K2 (previous lab). It classifies 32x32 RGB images into on of the CIFAR-10 classes. 

Input format: 
- 3 channels (RGB)
- Height: 32
- Width: 32
- FLattened into a list of 3072 float (C x H x W = 3 x 32 x 32)

Output:
- class_id (integer)
- label (string)

## Project structure
```
MLOPS25-M3-INTEGRATION-DISTRIBUTION/
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── params.yaml
├── README.md
├── .gitignore
├── data/
│   └── models/
        ├── model_weights.pt
│       └── model.ts.pt
├── scripts/
│   ├── export_torchscript.py
│   └── test_predict.py
└── src/
    ├── app/
    │   └── api.py
    └── ml/
        ├── config.py
        ├── model.py
        ├── predict.py
        ├── preprocess.py 
        └── utilis.py
```

### Step 1 - Export model to TorchScript
The trained Pytorch model from K2 was exported to TorchScript for deployment portability. 

Export command: 
`uv run python -m scripts.export_torchscript`
This generates: 
`data/models/models.ts.pt`

TorchScript was chosen because:
- It removies dependency on the original Python model definition.
- It allows deployment in production enviroments.
- It ensures model portability. 

### Step 2 - Model integration (FastAPI)
The model is loaded through a `Predictor`class:
`predictor = Predictor("params.yaml")`

The FastAPI endpoint:
```
@app.post("/predict")
def predict(req: PredictRequest):
return predictor.predict(req.x)
```

Request format:
```
{
    "x": [3072 float values]
}
````
The list represents a flattened RGB image in CxHxW order.

Response format:
```
{
    "class_id": 3, 
    "label": "cat"
}
```

### Step 3 - Running locally (without Docker)
Install dependencies:
`uv sync`
run API:
`uv run python -m src.app.api`
Open:
`http://localhost:8000/docs`

### Step 4 - Running with Docker
Build the container:
`docker build -t m3-api`

Run container: 
`docker run -p 8000:8000 m3-api`

Access:
`http://localhost:8000/docs`

## Testing the API
Reliable test using curl:
```
curl -s http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "$(python - << 'PY'
import json,random
x = [random.random() for _ in range(3072)]
print(json.dumps({"x": x}))
PY
)"
```
Expected: HTTP 200 with prediction JSON.

## Device handling (CPU vs MPS)
The model was exported and force to CPU during inference to ensure:
- Consistent Docker execution
- No dependency on Apple MPS
- No GPU assumptions in production
All tensors are explicity moved to CPU during export and inference.

This avoids device mismatch errors such as:
`input(device='cpu') and weight(device='mps=0') must be on the same device`

## Git workflow & collaboration
Development was performed using feature branches and pull requests.

Each feature was:
- Implemented in a dedicated branch
- Reviewed by a team member
- Merged into `main` only after approval

### Pull requests
#### PR 1 - FastAPI endpoint & validation
Implements: 
- POST /predict endpoint
- Pydantic request validation
- Proper HTTP error handling
Review included: 
- API contract clarity
- Swagger documentation improvements
- Structured JSON response format

Link: 
https://github.com/kevinbrunomlops/mlops25-m3-integration-distribution-/pull/2

#### PR 2 - Dockerization & Production readiness
Implements:
- Dockerfile
- Dependency locking via uv 
- Runtime validation inside container
Review included:
- Frozen dependency usage
- Container reproducibility
- Removal of dev dependencies from runtime

Link:
https://github.com/kevinbrunomlops/mlops25-m3-integration-distribution-/pull/3

## Submission chechlist
- ✅ Container builds succesfully
- ✅ Container starts and serves API
- ✅ POST /predict returns valid prediction
- ✅ README contains pull request links with documented code review

## Key MLOps concepts demonstrated
- Model export (TorchScript)
- Model encapsulation via predictor abstraction
- API-based model serving
- Input validation
- Containerized deployment
- Reproducible dependency managament
- Collaorative Git workflow
- Code review practices

## Conclussion
This project stimulates a production-style ML deployment workflow. 
Training (K2) -> Model export -> API integration -> Containerization -> Collaborative Review

The final result is a fully containerized inference service capable of classifying images via a REST API. 