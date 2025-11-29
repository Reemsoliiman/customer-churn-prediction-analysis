import sys
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
import plotly.graph_objects as go
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
VIZ_DIR = PROJECT_ROOT / "visualizations" / "interactive"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme matching notebook
SAGE_DARK = '#7A9B57'
NEUTRAL = '#F5F5DC'
PEACH_DARK = '#FF9A76'


def create_confusion_matrix_plot(y_test, y_pred, model_name, save_path):
    """
    Create and save confusion matrix visualization as HTML.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save HTML file
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Create interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn'],
        colorscale=[[0, SAGE_DARK], [0.5, NEUTRAL], [1, PEACH_DARK]],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        height=500,
        template="plotly_white",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    # Save to HTML
    fig.write_html(str(save_path))
    print(f"  Confusion matrix saved: {save_path.name}")
    
    return fig


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

        print("Evaluating models...")
        for name, filename in model_map.items():
            print(f"  Evaluating {name}...")
            model = joblib.load(MODELS_DIR / filename)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate comprehensive metrics
            metrics = {
                "model": name.replace("_", " ").title(),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob))
            }
            results.append(metrics)
            
            # Log metrics to MLflow (with model prefix for clarity)
            mlflow.log_metrics({
                f"{name}_accuracy": metrics["accuracy"],
                f"{name}_precision": metrics["precision"],
                f"{name}_recall": metrics["recall"],
                f"{name}_f1_score": metrics["f1_score"],
                f"{name}_roc_auc": metrics["roc_auc"]
            })

        # Find best model based on ROC-AUC
        best = max(results, key=lambda x: x["roc_auc"])
        best_filename = model_map[best["model"].lower().replace(" ", "_")]
        best_model = joblib.load(MODELS_DIR / best_filename)
        
        # Get predictions for best model to create confusion matrix
        best_y_pred = best_model.predict(X_test)
        
        # Create and save confusion matrix for best model
        cm_path = VIZ_DIR / "01_confusion_matrix.html"
        create_confusion_matrix_plot(y_test, best_y_pred, best["model"], cm_path)
        mlflow.log_artifact(str(cm_path), "visualizations")
        
        # Save best model
        joblib.dump(best_model, ARTIFACTS_DIR / "best_model_final.pkl")
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Log best model metrics separately
        mlflow.log_metrics({
            "best_model_accuracy": best["accuracy"],
            "best_model_precision": best["precision"],
            "best_model_recall": best["recall"],
            "best_model_f1_score": best["f1_score"],
            "best_model_roc_auc": best["roc_auc"]
        })
        mlflow.log_param("best_model_name", best["model"])

        # Save comprehensive evaluation results
        eval_results = {"best": best, "all": results}
        results_path = ARTIFACTS_DIR / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        mlflow.log_artifact(str(results_path), "results")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nBest Model: {best['model']}")
        print(f"  Accuracy:  {best['accuracy']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall:    {best['recall']:.4f}")
        print(f"  F1-Score:  {best['f1_score']:.4f}")
        print(f"  ROC-AUC:   {best['roc_auc']:.4f}")
        
        print("\nAll Models Performance:")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['model']:<20} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
                  f"{r['recall']:<10.4f} {r['f1_score']:<10.4f} {r['roc_auc']:<10.4f}")
        print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])