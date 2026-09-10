from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import read_config
from src.logger import logger

print("data_transformation.py loaded")


class DataTransformation:

    def __init__(self):

        print("1. Inside __init__")

        self.config = read_config()

        print("2. Config loaded")

        self.input_file = (
            Path(self.config["paths"]["processed_data"])
            / self.config["files"]["processed_csv"]
        )

        self.output_folder = Path(
            self.config["paths"]["transformed_data"]
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_file = (
            self.output_folder
            / self.config["files"]["transformed_csv"]
        )

        self.mapping_file = Path(
            "artifacts/label_mapping.json"
        )

        self.mapping_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def transform(self):

        print("3. Starting transformation...")
        logger.info("Starting Data Transformation")

        # ----------------------------------------
        # PASS 1 : Collect Labels
        # ----------------------------------------

        labels = set()

        print("4. Collecting labels...")

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):
            labels.update(chunk["Label"].dropna().unique())

        encoder = LabelEncoder()
        encoder.fit(sorted(labels))

        mapping = {
            label: int(code)
            for label, code in zip(
                encoder.classes_,
                encoder.transform(encoder.classes_)
            )
        }

        with open(self.mapping_file, "w") as f:
            json.dump(mapping, f, indent=4)

        print("5. Label mapping saved.")

        # ----------------------------------------
        # PASS 2 : Transform Dataset
        # ----------------------------------------

        columns_to_drop = [
            "Flow ID",
            "Src IP",
            "Dst IP",
            "Timestamp"
        ]

        total_rows = 0
        chunk_no = 0
        first_chunk = True

        print("6. Transforming dataset...")

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            chunk_no += 1

            # Drop unwanted columns
            chunk.drop(
                columns=columns_to_drop,
                errors="ignore",
                inplace=True
            )

            # Numeric columns
            numeric_cols = chunk.select_dtypes(
                include="number"
            ).columns

            # Fill missing values
            chunk[numeric_cols] = chunk[numeric_cols].fillna(
                chunk[numeric_cols].median()
            )

            # Replace infinity values
            chunk.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            # Fill NaN created from infinity replacement
            chunk[numeric_cols] = chunk[numeric_cols].fillna(
                chunk[numeric_cols].median()
            )

            # Fill any remaining NaN with 0
            chunk[numeric_cols] = chunk[numeric_cols].fillna(0)

            # Encode labels
            chunk["Label"] = encoder.transform(
                chunk["Label"]
            )

            # Save chunk
            chunk.to_csv(
                self.output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            total_rows += len(chunk)

            if chunk_no % 10 == 0:
                print(f"Processed {total_rows:,} rows...")
                logger.info(f"Processed {total_rows:,} rows")

        logger.info("Transformation completed.")

        print("\n====================================")
        print("Data Transformation Completed")
        print("====================================")
        print(f"Rows Processed : {total_rows:,}")
        print(f"Saved To : {self.output_file}")
        print(f"Label Mapping : {self.mapping_file}")


if __name__ == "__main__":

    transformer = DataTransformation()

    try:
        transformer.transform()
    except Exception as e:
        print("\nERROR:", e)
        logger.exception(e)