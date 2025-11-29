import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import uniform, randint
import mlflow
import mlflow.sklearn
import mlflow.xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


def quick_tune_model(base_model, param_dist, X_train, y_train, model_name):
    """Lightweight hyperparameter tuning with RandomizedSearchCV"""
    print(f"  Tuning {model_name}...")
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=8,  # Light: only 8 combinations
        cv=3,  # Light: only 3-fold CV
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"    Best score: {random_search.best_score_:.4f}")
    print(f"    Best params: {random_search.best_params_}")
    
    return random_search.best_estimator_, random_search.best_params_


def main(experiment_id: str):
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class balancing with SMOTE
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train.value_counts().to_dict()}")
    
    # Save test data
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_test, y_test), ARTIFACTS_DIR / "test_data.pkl")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define models with tuning parameters
    models_config = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                'C': uniform(0.1, 10),
                'penalty': ['l2']
            },
            "tune": True
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {},
            "tune": False  # Keep as baseline
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            "tune": True
        },
        "xgboost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.7, 0.3)
            },
            "tune": True
        }
    }
    
    # Train all models
    for name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print('='*60)
        
        with mlflow.start_run(run_name=name, nested=True, experiment_id=experiment_id):
            base_model = config["model"]
            
            # Hyperparameter tuning (if enabled)
            if config["tune"] and config["params"]:
                model, best_params = quick_tune_model(
                    base_model, config["params"], X_train, y_train, name
                )
                # Log best params
                for param_name, param_value in best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
            else:
                print(f"  Training {name} with default parameters...")
                model = base_model
                model.fit(X_train, y_train)
            
            # Cross-validation (5-fold)
            print(f"  Running 5-fold cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            print(f"    CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
            mlflow.log_metric("cv_roc_auc_mean", cv_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_std)
            
            # Final training on full training set (if not already trained)
            if config["tune"]:
                # Model already trained during tuning, just refit on full set
                model.fit(X_train, y_train)
            
            # Test set evaluation
            print(f"  Evaluating on test set...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            print(f"    Test Accuracy: {accuracy:.4f}")
            print(f"    Test ROC-AUC: {roc_auc:.4f}")
            
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_roc_auc", roc_auc)
            
            # Log model to MLflow
            if "xgb" in name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            
            model_path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)
            print(f"    Saved to {model_path.name}")
    
    print(f"\n{'='*60}")
    print("All models trained successfully!")
    print('='*60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <experiment_id>")
        sys.exit(1)
    main(sys.argv[1])