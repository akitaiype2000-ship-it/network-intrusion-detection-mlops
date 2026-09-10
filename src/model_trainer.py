from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.config import read_config
from src.logger import logger


class ModelTrainer:

    def __init__(self):

        self.config = read_config()

        split_path = Path(self.config["paths"]["split_data"])

        self.x_train_path = split_path / self.config["files"]["x_train"]
        self.y_train_path = split_path / self.config["files"]["y_train"]

        self.model_folder = Path(self.config["paths"]["model"])
        self.model_folder.mkdir(parents=True, exist_ok=True)

        self.model_path = (
            self.model_folder
            / self.config["files"]["model_file"]
        )

    def train(self):

        logger.info("Loading training data...")

        x_train = pd.read_csv(
            self.x_train_path,
            engine="python"
        )

        y_train = (
            pd.read_csv(
                self.y_train_path,
                engine="python"
            )
            .squeeze("columns")
        )

        logger.info(f"Training Samples : {len(x_train):,}")
        logger.info(f"Features : {x_train.shape[1]}")

        # Replace infinity values
        x_train.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Convert all columns to numeric
        x_train = x_train.apply(pd.to_numeric, errors="coerce")

        # Fill missing values with median
        x_train = x_train.fillna(
            x_train.median(numeric_only=True)
        )

        # Fill any remaining missing values with 0
        x_train = x_train.fillna(0)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        logger.info("Training Random Forest model...")

        model.fit(x_train, y_train)

        joblib.dump(model, self.model_path)

        logger.info("Model training completed.")
        logger.info(f"Model saved to {self.model_path}")

        print("\n===================================")
        print("Model Training Completed")
        print("===================================")
        print(f"Training Samples : {len(x_train):,}")
        print(f"Features         : {x_train.shape[1]}")
        print(f"Model Saved To   : {self.model_path}")


if __name__ == "__main__":

    trainer = ModelTrainer()
    trainer.train()