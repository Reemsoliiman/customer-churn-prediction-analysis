import sys
import pandas as pd
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from utils.helpers import handle_missing_values, encode_categorical_features

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "merged_churn_data.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

def main(experiment_id: str):
    with mlflow.start_run(run_name="preprocessing", experiment_id=experiment_id):
        print("Starting preprocessing...")

        df = pd.read_csv(RAW_PATH)
        mlflow.log_metric("raw_rows", len(df))

        df = df.drop_duplicates()
        df = handle_missing_values(df)
        df = encode_categorical_features(df)

        CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_PATH, index=False)

        mlflow.log_metric("cleaned_rows", len(df))
        mlflow.log_artifact(str(CLEANED_PATH), "cleaned_data")
        print("Preprocessing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])