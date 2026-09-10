from pathlib import Path
import joblib
import pandas as pd
import numpy as np
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

        self.scaler_path = Path(
            "artifacts/scaler.pkl"
        )

    def scale(self):

        logger.info("Loading selected dataset...")

        df = pd.read_csv(self.input_file)
        

        X = df.drop("Label", axis=1)
        y = df["Label"]

# Replace infinity values
        X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values
        X = X.fillna(X.median(numeric_only=True))

# Convert every column to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

# Fill any remaining NaNs
        X = X.fillna(0)

# Clip extremely large values
        X = X.clip(-1e10, 1e10)

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        

        scaled_df = pd.DataFrame(
            X_scaled,
            columns=X.columns
        )

        scaled_df["Label"] = y.values

        scaled_df.to_csv(
            self.output_file,
            index=False
        )

        joblib.dump(
            scaler,
            self.scaler_path
        )

        logger.info("Scaling completed.")

        print("\n===================================")
        print("Feature Scaling Completed")
        print("===================================")
        print(f"Rows : {len(df):,}")
        print(f"Saved : {self.output_file}")
        print(f"Scaler : {self.scaler_path}")


if __name__ == "__main__":

    scaler = FeatureScaling()

    scaler.scale()