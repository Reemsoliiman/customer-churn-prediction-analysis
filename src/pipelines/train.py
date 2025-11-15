import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

def main(experiment_id: str):
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_test, y_test), ARTIFACTS_DIR / "test_data.pkl")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, eval_metric='logloss', random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True, experiment_id=experiment_id):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            if "xgb" in name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            joblib.dump(model, MODELS_DIR / f"{name}.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])