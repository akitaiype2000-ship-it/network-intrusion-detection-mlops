from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import read_config
from src.logger import logger


class DataSplit:

    def __init__(self):

        self.config = read_config()

        self.input_file = (
            Path(self.config["paths"]["scaled_data"])
            / self.config["files"]["scaled_csv"]
        )

        self.output_folder = Path(
            self.config["paths"]["split_data"]
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.X_train_file = (
            self.output_folder
            / self.config["files"]["x_train"]
        )

        self.X_test_file = (
            self.output_folder
            / self.config["files"]["x_test"]
        )

        self.y_train_file = (
            self.output_folder
            / self.config["files"]["y_train"]
        )

        self.y_test_file = (
            self.output_folder
            / self.config["files"]["y_test"]
        )

    def split(self):

        logger.info("=" * 60)
        logger.info("DATA SPLITTING STARTED")
        logger.info("=" * 60)

        first_chunk = True

        total_rows = 0

        train_rows = 0

        test_rows = 0

        logger.info("Splitting dataset...")

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            X = chunk.drop(columns=["Label"])

            y = chunk["Label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )
            X_train.to_csv(
                self.X_train_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            X_test.to_csv(
                self.X_test_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            y_train.to_frame(name="Label").to_csv(
                self.y_train_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            y_test.to_frame(name="Label").to_csv(
                self.y_test_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            total_rows += len(chunk)
            train_rows += len(X_train)
            test_rows += len(X_test)

            if total_rows % 1000000 == 0:
                logger.info(
                    f"Processed {total_rows:,} rows..."
                )

        logger.info("=" * 60)
        logger.info("DATA SPLITTING COMPLETED")
        logger.info("=" * 60)

        print("\n===================================")
        print("Data Split Completed")
        print("===================================")
        print(f"Total Rows : {total_rows:,}")
        print(f"Train Rows : {train_rows:,}")
        print(f"Test Rows  : {test_rows:,}")
        print("===================================")
        
if __name__ == "__main__":

    DataSplit().split()