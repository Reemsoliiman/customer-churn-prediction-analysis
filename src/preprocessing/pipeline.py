"""
Data preprocessing pipeline: cleaning, imputation, encoding.
NO feature engineering here - that happens in feature_engineering.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger, log_dataframe_info, PipelineLogger
from src.utils.helpers import handle_missing_values, encode_categorical_features

logger = get_project_logger(__name__)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "merged_churn_data.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"


def main():
    """Run the preprocessing pipeline."""
    with PipelineLogger(logger, "Preprocessing Pipeline"):
        CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading raw data from: {RAW_PATH}")
        df = pd.read_csv(RAW_PATH)
        log_dataframe_info(logger, df, "Raw data")

        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        logger.info(f"Removed {n_removed} duplicate rows")

        logger.info("Handling missing values...")
        df = handle_missing_values(df)
        logger.info("Missing values handled")

        logger.info("Encoding categorical features...")
        df = encode_categorical_features(df)
        logger.info("Categorical features encoded")

        df.to_csv(CLEANED_PATH, index=False)
        log_dataframe_info(logger, df, "Cleaned data")

        logger.info(f"Cleaned data saved to: {CLEANED_PATH}")


if __name__ == "__main__":
    main()