"""
Merge churn-bigml-20.csv and churn-bigml-80.csv into a single dataset.
"""
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger, log_dataframe_info

logger = get_project_logger(__name__)


def collect_and_merge(file_20_path, file_80_path, output_path):
    try:
        logger.info(f"Reading {file_20_path}")
        df20 = pd.read_csv(file_20_path)
        log_dataframe_info(logger, df20, "df20")

        logger.info(f"Reading {file_80_path}")
        df80 = pd.read_csv(file_80_path)
        log_dataframe_info(logger, df80, "df80")

        logger.info("Merging datasets...")
        merged_data = pd.concat([df20, df80], axis=0, ignore_index=True)
        log_dataframe_info(logger, merged_data, "Merged data")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_data.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error during data collection and merging: {str(e)}")
        raise