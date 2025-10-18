import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_data_files(file_20_path, file_80_path):
    """
    Verify that the required data files exist in the specified paths.
    
    Args:
        file_20_path (str): Path to churn-bigml-20.csv
        file_80_path (str): Path to churn-bigml-80.csv
    
    Returns:
        list: Paths to the verified files
    
    Raises:
        FileNotFoundError: If any required file is missing
    """
    try:
        # Convert to Path objects for robust file handling
        file_20_path = Path(file_20_path)
        file_80_path = Path(file_80_path)
        
        # List of expected files
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
                    "Please download it manually from Kaggle (e.g., search for 'telecom churn dataset') "
                    "and place it in the data/raw/ directory."
                )
        
        return verified_files
    
    except Exception as e:
        logger.error(f"Error verifying data files: {str(e)}")
        raise

# if __name__ == "__main__":
#     # For testing the script independently
#     file_20_path = "data/raw/churn-bigml-20.csv"
#     file_80_path = "data/raw/churn-bigml-80.csv"
#     verified_files = verify_data_files(file_20_path, file_80_path)
#     print(f"Verified files: {verified_files}")