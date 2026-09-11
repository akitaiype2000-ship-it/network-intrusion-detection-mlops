from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.config import read_config
from src.logger import logger
class ModelEvaluation:

    def __init__(self):

        self.config = read_config()

        self.X_test_file = (
            Path(self.config["paths"]["split_data"])
            / self.config["files"]["x_test"]
        )

        self.y_test_file = (
            Path(self.config["paths"]["split_data"])
            / self.config["files"]["y_test"]
        )

        self.model_file = (
            Path(self.config["paths"]["model"])
            / self.config["files"]["model_file"]
        )

        self.report_file = (
            Path(self.config["paths"]["model"])
            / "evaluation_report.json"
        )
    def evaluate(self):

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION STARTED")
        logger.info("=" * 60)

        logger.info("Loading trained model...")

        model = joblib.load(
            self.model_file
        )

        y_true = []

        y_pred = []

        total_rows = 0

        logger.info("Evaluating model...")
        for X_chunk, y_chunk in zip(
            pd.read_csv(
                self.X_test_file,
                chunksize=100000,
                low_memory=False
            ),
            pd.read_csv(
                self.y_test_file,
                chunksize=100000,
                low_memory=False
            )
        ):

            y_chunk = y_chunk["Label"]

            predictions = model.predict(
                X_chunk
            )

            y_true.extend(
                y_chunk.tolist()
            )

            y_pred.extend(
                predictions.tolist()
            )

            total_rows += len(X_chunk)

            if total_rows % 1000000 == 0:

                logger.info(
                    f"Processed {total_rows:,} rows..."
                )
        logger.info("Computing evaluation metrics...")

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        report = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
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
        logger.info("MODEL EVALUATION COMPLETED")
        logger.info("=" * 60)

        print("\n===================================")
        print("Model Evaluation Completed")
        print("===================================")
        print(f"Rows Evaluated : {total_rows:,}")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")
        print(f"Report Saved   : {self.report_file}")
        print("===================================")
if __name__ == "__main__":

    ModelEvaluation().evaluate()