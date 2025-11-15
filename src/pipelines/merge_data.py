import sys
import pandas as pd
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main(experiment_id: str):
    with mlflow.start_run(run_name="merge_data", experiment_id=experiment_id):
        file_20 = Path(sys.argv[2])
        file_80 = Path(sys.argv[3])
        output_path = Path(sys.argv[4])

        df20 = pd.read_csv(file_20)
        df80 = pd.read_csv(file_80)
        merged = pd.concat([df20, df80], ignore_index=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

        mlflow.log_metric("rows_20", len(df20))
        mlflow.log_metric("rows_80", len(df80))
        mlflow.log_metric("rows_merged", len(merged))
        mlflow.log_artifact(str(output_path), "merged_data")
        print(f"Merged data saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_data.py <exp_id> <file20> <file80> <output>")
        sys.exit(1)
    main(sys.argv[1])