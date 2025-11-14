import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

# Load test data (already feature-selected, no scaling)
X_test, y_test = joblib.load(MODEL_DIR / "test_data.pkl")

model_files = {
    "logistic_regression": "logistic_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl"
}

results = []

for name, filename in model_files.items():
    model = joblib.load(MODEL_DIR / filename)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name.replace("_", " ").title(),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    results.append(metrics)

# Pick best by ROC-AUC
best = max(results, key=lambda x: x["roc_auc"])
best_model_name = best["model"].lower().replace(" ", "_")
best_filename = model_files[best_model_name]

# Load and save as generic final model
best_model = joblib.load(MODEL_DIR / best_filename)
final_model_path = MODEL_DIR / "best_model_final.pkl"
joblib.dump(best_model, final_model_path)

# Save results
eval_results = {
    "all_results": results,
    "best_model": best["model"],
    "best_auc": best["roc_auc"]
}
with open(MODEL_DIR / "evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\nEvaluation complete. Best model: {best['model']} (AUC: {best['roc_auc']:.4f})")
print(f"   → Final model saved: {final_model_path.name}")
print("Results saved to evaluation_results.json")