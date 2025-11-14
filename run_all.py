# run_all.py
import sys
import subprocess
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def run_script(script_path: str, description: str):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"{'='*60}")
    
    result = subprocess.run(["python", str(script_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {script_path}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description}")

def main():
    print("CHURN PREDICTION PIPELINE - FULL AUTOMATED RUN")
    print(f"Project Root: {PROJECT_ROOT}")

    run_script(PROJECT_ROOT / "src" / "data_collection" / "validate_dataset.py", "1. Validate raw data files")
    run_script(PROJECT_ROOT / "src" / "data_collection" / "collect_and_merge_data.py", "2. Merge datasets")
    run_script(PROJECT_ROOT / "src" / "preprocessing" / "pipeline.py", "3. Clean + basic features + scaling")
    run_script(PROJECT_ROOT / "src" / "preprocessing" / "feature_engineering.py", "4. Advanced features + RFE")
    run_script(PROJECT_ROOT / "src" / "modeling" / "train_model.py", "5. Train 4 models")
    run_script(PROJECT_ROOT / "src" / "modeling" / "evaluate.py", "6. Evaluate & save best model")

    # Final summary
    eval_path = PROJECT_ROOT / "models" / "trained_models" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        print(f"\nBEST MODEL: {data['best_model']} (AUC: {data['best_auc']:.4f})")
    else:
        print("\nEvaluation results not found.")

    print("\nALL DONE!")
    print("\nNext steps:")
    print("   streamlit run src/deployment/analyze_app.py")
    print("   streamlit run src/deployment/predict_app.py")

if __name__ == "__main__":
    main()