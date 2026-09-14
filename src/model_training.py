import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.linear_model import SGDClassifier

from src.config import read_config
from src.logger import logger
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Network Intrusion Detection")
class ModelTraining:

    def __init__(self):

        self.config = read_config()

        self.X_train_file = (
            Path(self.config["paths"]["split_data"])
            / self.config["files"]["x_train"]
        )

        self.y_train_file = (
            Path(self.config["paths"]["split_data"])
            / self.config["files"]["y_train"]
        )

        self.model_folder = Path(
            self.config["paths"]["model"]
        )

        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_file = (
            self.model_folder
            / self.config["files"]["model_file"]
        )

        self.report_file = (
            self.model_folder
            / "training_report.json"
        )
    def train(self):

        logger.info("=" * 60)
        logger.info("MODEL TRAINING STARTED")
        logger.info("=" * 60)

        logger.info("Loading class labels...")

        classes = pd.read_csv(
        self.y_train_file,
        usecols=["Label"]
    )["Label"].unique()

        logger.info(
        f"Number of Classes : {len(classes)}"
    )

        with mlflow.start_run():

            model = SGDClassifier(
                loss="log_loss",
                random_state=42
            )

            first_chunk = True
            total_rows = 0

            logger.info("Training model...")

            for X_chunk, y_chunk in zip(
                pd.read_csv(
                    self.X_train_file,
                    chunksize=100000,
                    low_memory=False
                ),
                pd.read_csv(
                    self.y_train_file,
                    chunksize=100000,
                    low_memory=False
                )
            ):

                y_chunk = y_chunk["Label"]

                if first_chunk:

                    model.partial_fit(
                        X_chunk,
                        y_chunk,
                        classes=classes
                    )

                    first_chunk = False

                else:

                    model.partial_fit(
                     X_chunk,
                        y_chunk
                    )

                total_rows += len(X_chunk)

                if total_rows % 1000000 == 0:

                    logger.info(
                        f"Processed {total_rows:,} rows..."
                    )

            logger.info("Saving trained model...")

            joblib.dump(
             model,
                self.model_file
            )

            mlflow.log_param("algorithm", "SGDClassifier")
            mlflow.log_param("loss", "log_loss")
            mlflow.log_param("training_rows", total_rows)
            mlflow.log_param("features", len(X_chunk.columns))
            mlflow.log_param("classes", len(classes))

            mlflow.sklearn.log_model(
                model,
                "model"
            )

            report = {
                "algorithm": "SGDClassifier",
                "loss": "log_loss",
                "training_rows": total_rows,
                "features": len(X_chunk.columns),
                "classes": len(classes),
                "model_path": str(self.model_file)
            }

            with open(
                    self.report_file,
                    "w"
                ) as file:

                    json.dump(
                        report,
                        file,
                        indent=4
                    )

        logger.info("=" * 60)
        logger.info("MODEL TRAINING COMPLETED")
        logger.info("=" * 60)

        print("\n===================================")
        print("Model Training Completed")
        print("===================================")
        print(f"Training Rows : {total_rows:,}")
        print(f"Classes       : {len(classes)}")
        print(f"Model Saved   : {self.model_file}")
        print(f"Report Saved  : {self.report_file}")
        print("===================================")
if __name__ == "__main__":

    ModelTraining().train()