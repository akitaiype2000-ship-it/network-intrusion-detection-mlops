from pathlib import Path
import pandas as pd

from src.logger import logger
from src.config import read_config


class DataIngestion:

    def __init__(self):
        self.config = read_config()

        self.raw_path = Path(self.config["paths"]["raw_data"])
        self.processed_path = Path(self.config["paths"]["processed_data"])

        self.output_file = (
            self.processed_path /
            self.config["files"]["processed_csv"]
        )

        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):

        csv_files = sorted(self.raw_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found.")

        logger.info(f"{len(csv_files)} CSV files found.")

        # Delete previous output if it exists
        if self.output_file.exists():
            self.output_file.unlink()

        total_rows = 0
        first_chunk = True

        for file in csv_files:

            logger.info(f"Processing {file.name}")

            chunk_number = 1

            try:

                for chunk in pd.read_csv(
                    file,
                    chunksize=100000,
                    low_memory=False
                ):

                    total_rows += len(chunk)

                    chunk.to_csv(
                        self.output_file,
                        mode="w" if first_chunk else "a",
                        header=first_chunk,
                        index=False
                    )

                    first_chunk = False

                    logger.info(
                        f"{file.name} | Chunk {chunk_number} | "
                        f"Rows: {len(chunk)} | "
                        f"Total: {total_rows}"
                    )

                    chunk_number += 1

            except Exception as e:
                logger.error(f"Error reading {file.name}: {e}")

        logger.info("=" * 50)
        logger.info("DATA INGESTION COMPLETED")
        logger.info(f"Total Rows Processed : {total_rows}")
        logger.info(f"Saved to : {self.output_file}")
        logger.info("=" * 50)

        print("\n====================================")
        print("Data Ingestion Completed Successfully")
        print("====================================")
        print(f"Total Rows Processed : {total_rows}")
        print(f"Saved to : {self.output_file}")

        return self.output_file


if __name__ == "__main__":

    ingestion = DataIngestion()
    ingestion.load_data()