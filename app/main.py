from fastapi import FastAPI

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

    result = predict(request.features)

    return {
        "prediction": result
    }