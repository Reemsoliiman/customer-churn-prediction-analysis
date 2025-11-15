import sys
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

def main(experiment_id: str):
    X_test, y_test = joblib.load(ARTIFACTS_DIR / "test_data.pkl")
    results = []

    with mlflow.start_run(run_name="evaluation", experiment_id=experiment_id):
        model_map = {
            "logistic_regression": "logistic_regression.pkl",
            "decision_tree": "decision_tree.pkl",
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl"
        }

        for name, filename in model_map.items():
            model = joblib.load(MODELS_DIR / filename)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = {
                "model": name.replace("_", " ").title(),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob))
            }
            results.append(metrics)

        best = max(results, key=lambda x: x["roc_auc"])
        best_filename = model_map[best["model"].lower().replace(" ", "_")]
        best_model = joblib.load(MODELS_DIR / best_filename)
        joblib.dump(best_model, ARTIFACTS_DIR / "best_model_final.pkl")
        mlflow.sklearn.log_model(best_model, "best_model")

        eval_results = {"best": best, "all": results}
        results_path = ARTIFACTS_DIR / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        mlflow.log_artifact(str(results_path), "results")

        print(f"BEST MODEL: {best['model']} (ROC-AUC: {best['roc_auc']:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])