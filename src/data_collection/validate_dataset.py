import sys
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main(experiment_id: str):
    with mlflow.start_run(run_name="validate_data", experiment_id=experiment_id):
        file_20 = Path(sys.argv[2])
        file_80 = Path(sys.argv[3])

        if not file_20.exists():
            raise FileNotFoundError(f"Missing: {file_20}")
        if not file_80.exists():
            raise FileNotFoundError(f"Missing: {file_80}")

        print(f"Found: {file_20.name}")
        print(f"Found: {file_80.name}")
        mlflow.log_metric("files_verified", 2)
        print("Validation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python validate_dataset.py <exp_id> <file20> <file80>")
        sys.exit(1)
    main(sys.argv[1])