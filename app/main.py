from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest
from app.predict import predict
from app.logger import logger

app = FastAPI(
    title="Network Intrusion Detection API",
    version="1.0"
)

@app.get("/")
def home():
    logger.info("Health check called")
    return {"message": "API is running"}

@app.post("/predict")
def predict_intrusion(request: PredictionRequest):

    if len(request.features) != 79:
        raise HTTPException(status_code=400, detail="Expected exactly 79 features.")

    prediction = predict(request.features)

    logger.info(f"Prediction: {prediction}")

    return {"prediction": prediction}