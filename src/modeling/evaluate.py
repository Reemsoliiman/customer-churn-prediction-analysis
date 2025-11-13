import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

# Load test data
X_test, y_test = joblib.load(MODEL_DIR / "test_data.pkl")
scaler = joblib.load(MODEL_DIR / "preprocessor.pkl")
X_test_scaled = scaler.transform(X_test)

# Model files
model_files = {
    "logistic_regression": "logistic_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl"
}

results = []

for name, filename in model_files.items():
    model = joblib.load(MODEL_DIR / filename)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "model": name.replace("_", " ").title(),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }
    results.append(metrics)

# Pick best by accuracy
best = max(results, key=lambda x: x["accuracy"])
best_model_name = best["model"]

# Save full results
import json
with open(MODEL_DIR / "evaluation_results.json", "w") as f:
    json.dump({"all_results": results, "best_model": best_model_name}, f)

print(f"\nEvaluation complete. Best model: {best_model_name} (AUC: {best['roc_auc']:.4f})")
print("Results saved to models/trained_models/evaluation_results.json")