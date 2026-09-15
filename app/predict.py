from pathlib import Path
import json
import joblib
import pandas as pd

ARTIFACTS = Path("artifacts")

model = joblib.load(ARTIFACTS / "model.pkl")
scaler = joblib.load(ARTIFACTS / "scaler.pkl")

with open(ARTIFACTS / "label_mapping.json", "r") as f:
    label_mapping = json.load(f)

reverse_mapping = {v: k for k, v in label_mapping.items()}


def predict(features: list[float]):

    df = pd.DataFrame([features])

    df = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )

    prediction = model.predict(df)[0]

    return reverse_mapping.get(prediction, prediction)