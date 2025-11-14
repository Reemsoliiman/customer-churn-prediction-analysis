"""
Train multiple classification models on the processed data.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger, log_dataframe_info, PipelineLogger

logger = get_project_logger(__name__)

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


def main():
    """Train all models."""
    with PipelineLogger(logger, "Model Training"):
        # Create directories
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load processed data
        logger.info(f"Loading processed data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        log_dataframe_info(logger, df, "Processed data")
        
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        
        # Ensure no missing values
        X = X.fillna(0)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Save test data to artifacts (for evaluation)
        test_data_path = ARTIFACTS_DIR / "test_data.pkl"
        joblib.dump((X_test, y_test), test_data_path)
        logger.info(f"âœ“ Test data saved to: {test_data_path}")
        
        # Define models
        models = {
            "logistic_regression": LogisticRegression(
                random_state=42, 
                max_iter=1000
            ),
            "decision_tree": DecisionTreeClassifier(
                random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300, 
                random_state=42,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        
        # Train and save each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save to trained_models directory
            model_path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  âœ“ Saved: {model_path}")
        
        logger.info(f"\nâœ“ All {len(models)} models trained and saved")


if __name__ == "__main__":
    main()
