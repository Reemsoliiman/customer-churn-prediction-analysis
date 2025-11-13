# run_all.py
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path (in case modules import each other)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def run_script(script_path: str, description: str):
    """Run a Python script and print status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        ["python", str(script_path)],
        cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        print(f"FAILED: {script_path}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description}")

def main():
    print("CHURN PREDICTION PIPELINE - FULL AUTOMATED RUN")
    print(f"Project Root: {PROJECT_ROOT}")

    # 1. Validate raw data
    run_script(
        PROJECT_ROOT / "src" / "data_collection" / "validate_dataset.py",
        "1. Validate raw data files"
    )

    # 2. Merge datasets
    run_script(
        PROJECT_ROOT / "src" / "data_collection" / "collect_and_merge_data.py",
        "2. Merge churn-bigml-20 and churn-bigml-80"
    )

    # 3. Clean & preprocess
    run_script(
        PROJECT_ROOT / "src" / "preprocessing" / "pipeline.py",
        "3. Clean data + basic feature engineering + scaling"
    )

    # 4. Advanced feature engineering + RFE
    run_script(
        PROJECT_ROOT / "src" / "preprocessing" / "feature_engineering.py",
        "4. Advanced features + RFE feature selection"
    )

    # 5. Train models
    run_script(
        PROJECT_ROOT / "src" / "modeling" / "train_model.py",
        "5. Train 4 models (LR, DT, RF, XGB)"
    )

    # 6. Evaluate models
    run_script(
        PROJECT_ROOT / "src" / "modeling" / "evaluate.py",
        "6. Evaluate models & save results"
    )

    # Final summary
    eval_path = PROJECT_ROOT / "models" / "trained_models" / "evaluation_results.json"
    if eval_path.exists():
        import json
        with open(eval_path) as f:
            data = json.load(f)
        best = data["best_model"]
        print(f"\nBEST MODEL: {best}")
    else:
        print("\nEvaluation results not found.")

    print("\nALL DONE!")
    print("\nNext steps:")
    print("   streamlit run src/deployment/analyze_app.py")
    print("   streamlit run src/deployment/predict_app.py")

if __name__ == "__main__":
    main()