"""
Validate that required raw data files exist.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger
logger = get_project_logger(__name__)


def verify_data_files(file_20_path, file_80_path):
    try:
        file_20_path = Path(file_20_path)
        file_80_path = Path(file_80_path)

        expected_files = [
            (file_20_path, 'churn-bigml-20.csv'),
            (file_80_path, 'churn-bigml-80.csv')
        ]

        verified_files = []
        for file_path, file_name in expected_files:
            if file_path.exists():
                logger.info(f"Found {file_name} at {file_path}")
                verified_files.append(str(file_path))
            else:
                logger.error(f"Missing {file_name} at {file_path}")
                raise FileNotFoundError(
                    f"{file_name} not found at {file_path}. "
                    "Please download it manually from Kaggle (search for 'telecom churn dataset') "
                    "and place it in the data/raw/ directory."
                )

        logger.info(f"All {len(verified_files)} required files verified successfully")
        return verified_files

    except Exception as e:
        logger.error(f"Error verifying data files: {str(e)}")
        raise