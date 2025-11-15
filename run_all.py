"""
Full automated churn-prediction pipeline.
Updated for new directory structure.
"""
import os
import sys
import subprocess
from pathlib import Path
import mlflow
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

mlruns_dir = PROJECT_ROOT / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri(mlruns_dir.as_uri())
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ----------------------------------------------------------------------
# 2. INITIALIZE MLFLOW EXPERIMENT
# ----------------------------------------------------------------------
EXPERIMENT_NAME = "ChurnPrediction-Pipeline"

def init_mlflow_experiment():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"Creating MLflow experiment: {EXPERIMENT_NAME}")
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"Experiment created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    
    return experiment_id


# ----------------------------------------------------------------------
# 3. HELPER: run script with PYTHONPATH
# ----------------------------------------------------------------------
def run_step(step_name: str, script_path: Path, experiment_id: str, *args):
    full_path = PROJECT_ROOT / script_path
    
    if not full_path.exists():
        print(f"ERROR: Script not found: {full_path}")
        sys.exit(1)
    
    print(f"\n[{step_name}] Running {full_path.name} ...")
    
    try:
        with mlflow.start_run(run_name=step_name, experiment_id=experiment_id):
            mlflow.log_artifact(str(full_path), "scripts")

            env = os.environ.copy()
            python_path = env.get("PYTHONPATH", "")
            src_path = str(SRC_DIR)
            env["PYTHONPATH"] = src_path + os.pathsep + python_path

            cmd = [sys.executable, str(full_path), experiment_id] + [str(a) for a in args]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                shell=True,
                env=env
            )

            mlflow.log_text(result.stdout or "No output", "stdout.txt")
            if result.stderr:
                mlflow.log_text(result.stderr, "stderr.txt")

            if result.returncode != 0:
                mlflow.set_tag("status", "FAILED")
                print(f"{step_name} FAILED")
                print(f"STDERR: {result.stderr}")
                sys.exit(1)
            else:
                mlflow.set_tag("status", "SUCCESS")
                print(f"{step_name} SUCCESS")
    except Exception as e:
        print(f"{step_name} MLflow error: {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# 4. MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CHURN PREDICTION PIPELINE - FULL AUTOMATED RUN")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 60)

    experiment_id = init_mlflow_experiment()

    # ------------------------------------------------------------------
    # 1. Validate raw files
    # ------------------------------------------------------------------
    raw_20 = PROJECT_ROOT / "data" / "raw" / "churn-bigml-20.csv"
    raw_80 = PROJECT_ROOT / "data" / "raw" / "churn-bigml-80.csv"
    
    if not raw_20.exists() or not raw_80.exists():
        print("ERROR: Raw data files not found!")
        print(f"Expected: {raw_20} and {raw_80}")
        print("Download from Kaggle: 'telecom churn dataset'")
        sys.exit(1)
    
    run_step(
        "1/6 Validate data",
        Path("src/pipelines/validate_data.py"),
        experiment_id,
        str(raw_20),
        str(raw_80)
    )

    # ------------------------------------------------------------------
    # 2. Merge raw CSVs
    # ------------------------------------------------------------------
    merged_path = PROJECT_ROOT / "data" / "raw" / "merged_churn_data.csv"
    run_step(
        "2/6 Merge raw data",
        Path("src/pipelines/merge_data.py"),
        experiment_id,
        str(raw_20),
        str(raw_80),
        str(merged_path)
    )

    # ------------------------------------------------------------------
    # 3. Pre-process
    # ------------------------------------------------------------------
    run_step(
        "3/6 Pre-processing",
        Path("src/pipelines/preprocess.py"),
        experiment_id
    )

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    run_step(
        "4/6 Feature engineering",
        Path("src/pipelines/engineer_features.py"),
        experiment_id
    )

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    run_step(
        "5/6 Train models",
        Path("src/pipelines/train.py"),
        experiment_id
    )

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    run_step(
        "6/6 Evaluation",
        Path("src/pipelines/evaluate.py"),
        experiment_id
    )

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("Open MLflow UI:")
    print(f"    cd \"{PROJECT_ROOT}\"")
    print("    mlflow ui")
    print("    Open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()