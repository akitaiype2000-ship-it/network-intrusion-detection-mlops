from pathlib import Path
import json
import pandas as pd

from src.logger import logger
from src.config import read_config


class DataValidation:

    def __init__(self):
        config = read_config()

        self.data_path = (
            Path(config["paths"]["processed_data"])
            / config["files"]["processed_csv"]
        )

        self.report_path = Path("artifacts/validation_report.json")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def validate(self):

        logger.info("========== DATA VALIDATION STARTED ==========")

        # Check dataset exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        logger.info(f"Loading dataset from {self.data_path}")

        df = pd.read_csv(
            self.data_path,
            nrows=100000,          # Validate first 100k rows
            low_memory=False
        )

        logger.info("Dataset loaded successfully.")

        # Dataset should not be empty
        if df.empty:
            raise ValueError("Dataset is empty.")

        # Target column must exist
        if "Label" not in df.columns:
            raise ValueError("Target column 'Label' not found.")

        # Validation statistics
        rows = len(df)
        columns = len(df.columns)

        missing_values = df.isnull().sum().to_dict()
        total_missing = int(df.isnull().sum().sum())

        duplicate_rows = int(df.duplicated().sum())

        dtypes = {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        }

        labels = (
            df["Label"]
            .value_counts()
            .to_dict()
        )

        num_classes = int(df["Label"].nunique())

        # Validation report
        report = {
            "status": "Passed",
            "rows_checked": rows,
            "columns": columns,
            "total_missing_values": total_missing,
            "missing_values_per_column": missing_values,
            "duplicate_rows": duplicate_rows,
            "number_of_classes": num_classes,
            "label_distribution": labels,
            "data_types": dtypes
        }

        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info("Validation report saved successfully.")
        logger.info("========== DATA VALIDATION COMPLETED ==========")

        # Console summary
        print("\n====================================")
        print(" DATA VALIDATION COMPLETED")
        print("====================================")
        print(f"Rows Checked         : {rows:,}")
        print(f"Columns              : {columns}")
        print(f"Missing Values       : {total_missing:,}")
        print(f"Duplicate Rows       : {duplicate_rows:,}")
        print(f"Number of Classes    : {num_classes}")
        print(f"Validation Status    : PASSED")
        print(f"Report Saved To      : {self.report_path}")
        print("====================================")


if __name__ == "__main__":

    validator = DataValidation()
    validator.validate()