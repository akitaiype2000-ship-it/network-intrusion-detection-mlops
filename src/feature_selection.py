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

        logger.info("=" * 60)
        logger.info("FEATURE SELECTION STARTED")
        logger.info("=" * 60)

        # ----------------------------------------
        # PASS 1 : Detect constant columns
        # ----------------------------------------

        logger.info("Pass 1 : Scanning dataset...")

        first_chunk = True
        candidate_constants = {}
        total_rows = 0

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            total_rows += len(chunk)

            if first_chunk:

                for col in chunk.columns:

                    candidate_constants[col] = (
                        chunk[col].nunique(dropna=False) <= 1
                    )

                first_chunk = False

            else:

                for col in chunk.columns:

                    if candidate_constants[col]:

                        if chunk[col].nunique(dropna=False) > 1:

                            candidate_constants[col] = False

            if total_rows % 1000000 == 0:

                logger.info(
                    f"Scanned {total_rows:,} rows..."
                )

        constant_columns = [

            col

            for col, constant in candidate_constants.items()

            if constant and col != "Label"

        ]

        logger.info(
            f"Constant Columns Found : {len(constant_columns)}"
        )

        # ----------------------------------------
        # PASS 2 : Write cleaned dataset
        # ----------------------------------------

        logger.info("Pass 2 : Writing cleaned dataset...")

        first_chunk = True
        total_rows = 0

        original_features = None
        selected_features = None
        final_feature_names = None

        for chunk in pd.read_csv(
            self.input_file,
            chunksize=100000,
            low_memory=False
        ):

            if original_features is None:

                original_features = len(chunk.columns)

            chunk.drop(
                columns=constant_columns,
                inplace=True,
                errors="ignore"
            )

            if selected_features is None:

                selected_features = len(chunk.columns)

                final_feature_names = list(chunk.columns)

            chunk.to_csv(
                self.output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

            total_rows += len(chunk)

            if total_rows % 1000000 == 0:

                logger.info(
                    f"Written {total_rows:,} rows..."
                )

        # ----------------------------------------
        # Save report
        # ----------------------------------------

        report = {

            "rows_processed": total_rows,

            "original_features": original_features,

            "selected_features": selected_features,

            "removed_constant_columns": constant_columns,

            "final_features": final_feature_names

        }

        with open(
            self.report_file,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        logger.info("=" * 60)
        logger.info("FEATURE SELECTION COMPLETED")
        logger.info("=" * 60)

        print("\n====================================")
        print("FEATURE SELECTION COMPLETED")
        print("====================================")
        print(f"Rows Processed            : {total_rows:,}")
        print(f"Original Features         : {original_features}")
        print(f"Selected Features         : {selected_features}")
        print(f"Removed Constant Columns  : {len(constant_columns)}")
        print(f"Saved To                  : {self.output_file}")
        print(f"Report                    : {self.report_file}")
        print("====================================")


if __name__ == "__main__":

    FeatureSelection().select_features()