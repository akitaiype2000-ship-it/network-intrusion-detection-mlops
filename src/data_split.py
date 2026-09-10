from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import read_config
from src.logger import logger


class DataSplit:

    def __init__(self):

        self.config = read_config()

        # Input dataset
        self.input_file = (
            Path(self.config["paths"]["featured_data"])
            / self.config["files"]["selected_csv"]
        )

        # Output folder
        self.output_folder = Path(
            self.config["paths"]["split_data"]
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

    def split(self):

        logger.info("Loading selected dataset...")

        df = pd.read_csv(
            self.input_file,
            low_memory=False
        )

        logger.info(f"Dataset Shape : {df.shape}")

        # Features and Target
        X = df.drop(columns=["Label"])
        y = df["Label"]

        # Train-Test Split
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Save split datasets
        x_train.to_csv(
            self.output_folder / self.config["files"]["x_train"],
            index=False
        )

        x_test.to_csv(
            self.output_folder / self.config["files"]["x_test"],
            index=False
        )

        y_train.to_frame().to_csv(
            self.output_folder / self.config["files"]["y_train"],
            index=False
        )

        y_test.to_frame().to_csv(
            self.output_folder / self.config["files"]["y_test"],
            index=False
        )

        logger.info("Train-Test Split Completed")

        print("\n===================================")
        print("Train-Test Split Completed")
        print("===================================")
        print(f"Training Samples : {len(x_train):,}")
        print(f"Testing Samples  : {len(x_test):,}")
        print(f"Features         : {x_train.shape[1]}")
        print(f"Saved To         : {self.output_folder}")


if __name__ == "__main__":

    splitter = DataSplit()

    splitter.split()