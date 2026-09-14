from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation
from src.feature_selection import FeatureSelection
from src.feature_scaling import FeatureScaling
from src.data_split import DataSplit
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation
from src.logger import logger


def run_pipeline():

    print("\n===================================")
    print("TRAINING PIPELINE STARTED")
    print("===================================\n")

    logger.info("Starting Training Pipeline")

    # Step 1
    print("Step 1: Data Ingestion")
    DataIngestion().load_data()
    print("✓ Completed\n")

    # Step 2
    print("Step 2: Data Validation")
    DataValidation().validate()
    print("✓ Completed\n")

    # Step 3
    print("Step 3: Data Transformation")
    DataTransformation().transform()
    print("✓ Completed\n")

    # Step 4
    print("Step 4: Feature Selection")
    FeatureSelection().select_features()
    print("✓ Completed\n")

    # Step 5
    print("Step 5: Feature Scaling")
    FeatureScaling().scale()
    print("✓ Completed\n")

    # Step 6
    print("Step 6: Train/Test Split")
    DataSplit().split()
    print("✓ Completed\n")

    # Step 7
    print("Step 7: Model Training")
    ModelTraining().train()
    print("✓ Completed\n")

    # Step 8
    print("Step 8: Model Evaluation")
    ModelEvaluation().evaluate()
    print("✓ Completed\n")

    logger.info("Training Pipeline Completed")

    print("\n===================================")
    print("TRAINING PIPELINE COMPLETED")
    print("===================================")


if __name__ == "__main__":
    run_pipeline()