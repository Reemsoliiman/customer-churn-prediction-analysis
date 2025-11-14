"""
Evaluate all trained models and select the best one.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_project_logger, log_model_metrics, PipelineLogger

logger = get_project_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


def main():
    """Evaluate all models and save the best one."""
    with PipelineLogger(logger, "Model Evaluation"):
        test_data_path = ARTIFACTS_DIR / "test_data.pkl"
        logger.info(f"Loading test data from: {test_data_path}")
        X_test, y_test = joblib.load(test_data_path)
        logger.info(f"Test set size: {X_test.shape}")

        model_files = {
            "logistic_regression": "logistic_regression.pkl",
            "decision_tree": "decision_tree.pkl",
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl"
        }

        results = []

        for name, filename in model_files.items():
            logger.info(f"\nEvaluating {name}...")

            model_path = MODELS_DIR / filename
            model = joblib.load(model_path)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name.replace("_", " ").title(),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

            results.append(metrics)
            log_model_metrics(logger, metrics.get("model", ""), {k: v for k, v in metrics.items() if k != "model"})

            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

        best = max(results, key=lambda x: x["roc_auc"])
        best_model_name = best["model"].lower().replace(" ", "_")
        best_filename = model_files[best_model_name]

        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best['model']}")
        logger.info(f"ROC-AUC: {best['roc_auc']:.4f}")
        logger.info(f"{'='*60}")

        best_model = joblib.load(MODELS_DIR / best_filename)
        final_model_path = ARTIFACTS_DIR / "best_model_final.pkl"
        joblib.dump(best_model, final_model_path)
        logger.info(f"Best model saved to: {final_model_path}")

        eval_results = {
            "all_results": results,
            "best_model": best["model"],
            "best_model_filename": best_filename,
            "best_metrics": best
        }

        results_path = ARTIFACTS_DIR / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Evaluation results saved to: {results_path}")

        results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
        logger.info("\n=== MODEL COMPARISON ===")
        logger.info(f"\n{results_df.to_string(index=False)}")


if __name__ == "__main__":
    main()