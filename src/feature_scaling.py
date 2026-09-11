from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import read_config
from src.logger import logger


class FeatureScaling:

    def __init__(self):

        self.config = read_config()

        self.input_file = (
            Path(self.config["paths"]["featured_data"])
            / self.config["files"]["selected_csv"]
        )

        self.output_folder = Path(
            self.config["paths"]["scaled_data"]
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_file = (
            self.output_folder
            / self.config["files"]["scaled_csv"]
        )

        self.scaler_path = Path("artifacts/scaler.pkl")

        self.scaler_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def clean_chunk(self, X):

        X = X.apply(pd.to_numeric, errors="coerce")

        X = X.replace([np.inf, -np.inf], np.nan)

        max_float = np.finfo(np.float64).max

        X = X.clip(
            lower=-max_float,
            upper=max_float
        )

        X = X.fillna(
            X.median(numeric_only=True)
        )

        X = X.fillna(0)

        return X

    def scale(self):

        logger.info("=" * 60)
        logger.info("FEATURE SCALING STARTED")
        logger.info("=" * 60)

        scaler = StandardScaler()

        logger.info(
            "Pass 1 : Computing scaling parameters..."
        )

        total_rows = 0

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            X = chunk.drop(columns=["Label"])

            X = self.clean_chunk(X)

            scaler.partial_fit(X)

            total_rows += len(chunk)

            if total_rows % 1000000 == 0:

                logger.info(
                    f"Processed {total_rows:,} rows..."
                )

        logger.info(
            f"Scaling parameters learned from {total_rows:,} rows."
        )

        logger.info(
            "Pass 2 : Scaling dataset..."
        )

        first_chunk = True
        total_rows = 0

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            y = chunk["Label"]

            X = chunk.drop(columns=["Label"])

            X = self.clean_chunk(X)

            X_scaled = scaler.transform(X)

            scaled_chunk = pd.DataFrame(
                X_scaled,
                columns=X.columns
            )

            scaled_chunk["Label"] = y.values

            scaled_chunk.to_csv(
                self.output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            total_rows += len(chunk)

            if total_rows % 1000000 == 0:

                logger.info(
                    f"Scaled {total_rows:,} rows..."
                )

        joblib.dump(
            scaler,
            self.scaler_path
        )

        logger.info("=" * 60)
        logger.info("FEATURE SCALING COMPLETED")
        logger.info("=" * 60)

        print("\n===================================")
        print("Feature Scaling Completed")
        print("===================================")
        print(f"Rows Scaled : {total_rows:,}")
        print(f"Saved To    : {self.output_file}")
        print(f"Scaler      : {self.scaler_path}")
        print("===================================")


if __name__ == "__main__":

    FeatureScaling().scale()