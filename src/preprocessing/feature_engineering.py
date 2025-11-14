"""
Feature engineering and selection using RFE.
Uses shared helper functions to ensure consistency with deployment.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger, log_dataframe_info, PipelineLogger
from src.utils.helpers import engineer_features

logger = get_project_logger(__name__)

CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
FINAL_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
FEATS_PATH = ARTIFACTS_DIR / "selected_features.pkl"


def main():
    """Run feature engineering and selection."""
    with PipelineLogger(logger, "Feature Engineering & Selection"):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading cleaned data from: {CLEANED_PATH}")
        df = pd.read_csv(CLEANED_PATH)
        log_dataframe_info(logger, df, "Cleaned data")

        y = df["Churn"].astype(int)
        X_raw = df.drop("Churn", axis=1)

        logger.info(f"Original features: {X_raw.shape[1]}")

        logger.info("Creating engineered features...")
        X = engineer_features(X_raw, is_training=True)
        logger.info(f"Features after engineering: {X.shape[1]}")
        log_dataframe_info(logger, X, "Engineered features")

        logger.info("Running RFE (selecting 20 features)...")
        rfe = RFE(
            RandomForestClassifier(n_estimators=200, random_state=42),
            n_features_to_select=20,
            verbose=1
        )
        rfe.fit(X, y)

        selected = X.columns[rfe.support_].tolist()
        logger.info(f"Selected {len(selected)} features")
        logger.info(f"Selected features: {selected}")

        df_final = X[selected].copy()
        df_final["Churn"] = y
        df_final.to_csv(FINAL_PATH, index=False)
        logger.info(f"Final data saved to: {FINAL_PATH}")

        joblib.dump(selected, FEATS_PATH)
        logger.info(f"Selected features saved to: {FEATS_PATH}")

        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'selected': rfe.support_,
            'ranking': rfe.ranking_
        }).sort_values('ranking')

        logger.info("\nTop 10 features by ranking:")
        logger.info(f"\n{feature_ranking.head(10).to_string()}")


if __name__ == "__main__":
    main()