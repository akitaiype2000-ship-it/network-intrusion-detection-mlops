from pathlib import Path
import json

import pandas as pd

from src.config import read_config
from src.logger import logger


class FeatureSelection:

    def __init__(self):

        self.config = read_config()

        self.input_file = (
            Path(self.config["paths"]["transformed_data"])
            / self.config["files"]["transformed_csv"]
        )

        self.output_folder = Path(
            self.config["paths"]["featured_data"]
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_file = (
            self.output_folder
            / self.config["files"]["selected_csv"]
        )

        self.report_file = Path(
            "artifacts/feature_selection_report.json"
        )

        self.report_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def select_features(self):

        logger.info("Starting Feature Selection...")

        df = pd.read_csv(
            self.input_file,
            low_memory=False
        )

        original_columns = list(df.columns)

        # -------------------------------
        # Remove constant columns
        # -------------------------------

        constant_columns = [
            col
            for col in df.columns
            if df[col].nunique() <= 1
        ]

        df.drop(
            columns=constant_columns,
            inplace=True,
            errors="ignore"
        )

        # -------------------------------
        # Remove duplicate columns
        # -------------------------------

        duplicate_columns = []

        cols = df.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_columns.append(cols[j])

        df.drop(
            columns=duplicate_columns,
            inplace=True,
            errors="ignore"
        )

        # -------------------------------
        # Save cleaned dataset
        # -------------------------------

        df.to_csv(
            self.output_file,
            index=False
        )

        report = {
            "original_features": len(original_columns),
            "selected_features": len(df.columns),
            "removed_constant_columns": constant_columns,
            "removed_duplicate_columns": duplicate_columns,
            "final_features": list(df.columns)
        }

        with open(self.report_file, "w") as f:
            json.dump(report, f, indent=4)

        logger.info("Feature Selection Completed.")

        print("\n===================================")
        print("Feature Selection Completed")
        print("===================================")
        print(f"Original Features : {len(original_columns)}")
        print(f"Selected Features : {len(df.columns)}")
        print(f"Removed Constant Columns : {len(constant_columns)}")
        print(f"Removed Duplicate Columns : {len(duplicate_columns)}")
        print(f"Saved To : {self.output_file}")
        print(f"Report : {self.report_file}")


if __name__ == "__main__":

    selector = FeatureSelection()

    selector.select_features()