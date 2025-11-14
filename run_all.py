# run_all.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import functions directly instead of subprocess
from src.data_collection.validate_dataset import verify_data_files
from src.data_collection.collect_and_merge_data import collect_and_merge
from src.preprocessing.pipeline import main as pipeline_main
from src.preprocessing.feature_engineering import main as feature_eng_main
from src.modeling.train_model import main as train_main
from src.modeling.evaluate import main as evaluate_main

def main():
    print("="*60)
    print("CHURN PREDICTION PIPELINE - FULL AUTOMATED RUN")
    print(f"Project Root: {PROJECT_ROOT}")
    print("="*60)

    try:
        # Step 1: Validate raw data
        print("\n[1/6] Validating raw data files...")
        verify_data_files(
            PROJECT_ROOT / "data/raw/churn-bigml-20.csv",
            PROJECT_ROOT / "data/raw/churn-bigml-80.csv"
        )
        print("âœ“ Data files validated")

        # Step 2: Merge datasets
        print("\n[2/6] Merging datasets...")
        collect_and_merge(
            PROJECT_ROOT / "data/raw/churn-bigml-20.csv",
            PROJECT_ROOT / "data/raw/churn-bigml-80.csv",
            PROJECT_ROOT / "data/raw/merged_churn_data.csv"
        )
        print("âœ“ Datasets merged")

        # Step 3: Basic preprocessing
        print("\n[3/6] Running preprocessing pipeline...")
        pipeline_main()
        print("âœ“ Preprocessing complete")

        # Step 4: Feature engineering + selection
        print("\n[4/6] Feature engineering + RFE selection...")
        feature_eng_main()
        print("âœ“ Feature engineering complete")

        # Step 5: Train models
        print("\n[5/6] Training models...")
        train_main()
        print("âœ“ Model training complete")

        # Step 6: Evaluate and select best
        print("\n[6/6] Evaluating models...")
        evaluate_main()
        print("âœ“ Evaluation complete")

        print("\n" + "="*60)
        print("ðŸŽ‰ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  streamlit run src/deployment/streamlit_app.py")

    except Exception as e:
        print(f"\nâŒ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
