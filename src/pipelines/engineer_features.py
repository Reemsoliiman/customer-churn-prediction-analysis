import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import mlflow
from src.utils.helpers import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
FINAL_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
FEATS_PATH = PROJECT_ROOT / "models" / "artifacts" / "selected_features.pkl"

def main(experiment_id: str):
    with mlflow.start_run(run_name="feature_engineering", experiment_id=experiment_id):
        df = pd.read_csv(CLEANED_PATH)
        y = df["Churn"].astype(int)
        X = engineer_features(df.drop("Churn", axis=1))

        rfe = RFE(RandomForestClassifier(n_estimators=200, random_state=42), n_features_to_select=20)
        rfe.fit(X, y)
        selected = X.columns[rfe.support_].tolist()

        final_df = X[selected].copy()
        final_df["Churn"] = y
        FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(FINAL_PATH, index=False)

        FEATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selected, FEATS_PATH)

        mlflow.log_metric("selected_features", len(selected))
        mlflow.log_artifact(str(FEATS_PATH), "features")
        mlflow.log_artifact(str(FINAL_PATH), "final_data")
        print(f"Selected {len(selected)} features")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python engineer_features.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])