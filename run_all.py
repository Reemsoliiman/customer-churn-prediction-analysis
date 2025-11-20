"""
Enhanced pipeline orchestrator with integrated monitoring.
Extends run_all.py with monitoring and retraining capabilities.
"""
import os
import sys
import subprocess
from pathlib import Path
import mlflow
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

mlruns_dir = PROJECT_ROOT / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri(mlruns_dir.as_uri())
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


EXPERIMENT_NAME = "ChurnPrediction-Pipeline"


def init_mlflow_experiment():
    """Initialize or get existing MLflow experiment"""
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


def run_step(step_name: str, script_path: Path, experiment_id: str, *args):
    """Execute a pipeline step with MLflow tracking"""
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


def main():
    print("=" * 60)
    print("CHURN PREDICTION PIPELINE WITH MONITORING")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 60)

    experiment_id = init_mlflow_experiment()

    # ------------------------------------------------------------------
    # PART 1: INITIAL TRAINING PIPELINE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 1: INITIAL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Validate raw files
    raw_20 = PROJECT_ROOT / "data" / "raw" / "churn-bigml-20.csv"
    raw_80 = PROJECT_ROOT / "data" / "raw" / "churn-bigml-80.csv"
    
    if not raw_20.exists() or not raw_80.exists():
        print("ERROR: Raw data files not found!")
        print(f"Expected: {raw_20} and {raw_80}")
        print("Download from Kaggle: 'telecom churn dataset'")
        sys.exit(1)
    
    run_step(
        "1/7 Validate data",
        Path("src/pipelines/validate_data.py"),
        experiment_id,
        str(raw_20),
        str(raw_80)
    )

    # 2. Merge raw CSVs
    merged_path = PROJECT_ROOT / "data" / "raw" / "merged_churn_data.csv"
    run_step(
        "2/7 Merge raw data",
        Path("src/pipelines/merge_data.py"),
        experiment_id,
        str(raw_20),
        str(raw_80),
        str(merged_path)
    )

    # 3. Pre-process
    run_step(
        "3/7 Pre-processing",
        Path("src/pipelines/preprocess.py"),
        experiment_id
    )

    # 4. Feature engineering
    run_step(
        "4/7 Feature engineering",
        Path("src/pipelines/engineer_features.py"),
        experiment_id
    )

    # 5. Train models
    run_step(
        "5/7 Train models",
        Path("src/pipelines/train.py"),
        experiment_id
    )

    # 6. Evaluate
    run_step(
        "6/7 Evaluation",
        Path("src/pipelines/evaluate.py"),
        experiment_id
    )

    # ------------------------------------------------------------------
    # PART 2: MONITORING AND RETRAINING
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 2: MONITORING AND RETRAINING")
    print("=" * 60)

    # 7. Monitor performance
    run_step(
        "7/7 Monitor performance",
        Path("src/pipelines/run_monitoring.py"),
        experiment_id
    )

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. View training results:")
    print("   mlflow ui")
    print("   Open: http://localhost:5000")
    print()
    print("2. Run monitoring cycle:")
    print(f"   python src/pipelines/run_monitoring.py {experiment_id}")
    print()
    print("3. Trigger retraining (if needed):")
    print(f"   python src/pipelines/trigger_retraining.py {experiment_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()