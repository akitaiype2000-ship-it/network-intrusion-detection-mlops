from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import PredictionRequest
from app.predict import predict
from app.logger import logger

app = FastAPI(
    title="Network Intrusion Detection API",
    description="""
A production-ready machine learning API for detecting network intrusions using a trained SGDClassifier.

## Features
- Predict network traffic labels
- Health check endpoint
- Dockerized deployment
- CI/CD with GitHub Actions
- Hosted on AWS EC2
""",
    version="1.0.0",
    contact={
        "name": "Aku",
        "email": "your-email@example.com"
    },
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    logger.info("Health check called")
    return {"message": "API is running"}
@app.get("/health")
def health():
    logger.info("Health endpoint called")
    return {
        "status": "healthy"
    }
@app.post("/predict")
def predict_intrusion(request: PredictionRequest):

    if len(request.features) != 79:
        raise HTTPException(status_code=400, detail="Expected exactly 79 features.")

    prediction = predict(request.features)

    logger.info(f"Prediction: {prediction}")

    return {"prediction": prediction}

