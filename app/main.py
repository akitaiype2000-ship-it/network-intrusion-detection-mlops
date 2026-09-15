from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest
from app.predict import predict

app = FastAPI(
    title="Network Intrusion Detection API",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict_intrusion(request: PredictionRequest):

    if len(request.features) != 79:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 79 features."
        )

    result = predict(request.features)

    return {
        "prediction": result
    }