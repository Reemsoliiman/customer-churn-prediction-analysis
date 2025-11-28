"""
Automated retraining pipeline integrated with MLflow.
Follows the same structure as existing training pipeline in src/pipelines/.
"""
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.email_alerts import get_alerter


class RetrainingPipeline:
    """
    Automated retraining pipeline with MLflow integration.
    Uses same approach as src/pipelines/train.py
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models" / "trained_models"
        self.artifacts_dir = self.project_root / "models" / "artifacts"
        
        # MLflow setup
        mlflow.set_tracking_uri((self.project_root / "mlruns").as_uri())
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality before retraining.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, message)
        """
        issues = []
        
        if len(df) < 1000:
            issues.append(f"Insufficient samples: {len(df)} < 1000")
        
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.1:
            issues.append(f"Excessive missing values: {missing_pct:.1%}")
        
        if 'Churn' in df.columns:
            churn_rate = df['Churn'].mean()
            if churn_rate < 0.05 or churn_rate > 0.5:
                issues.append(f"Unusual churn rate: {churn_rate:.1%}")
        
        is_valid = len(issues) == 0
        message = "Data validation passed" if is_valid else "; ".join(issues)
        
        return is_valid, message
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Train models using same configuration as original pipeline.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            experiment_id: MLflow experiment ID
            
        Returns:
            Dictionary with training results
        """
        print("Training models...")
        
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                eval_metric='logloss',
                random_state=42
            )
        }
        
        results = {}
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            with mlflow.start_run(
                run_name=f"retrain_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                nested=True,
                experiment_id=experiment_id
            ):
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0)
                }
                
                # Log to MLflow
                mlflow.log_params({
                    'model_type': name,
                    'retrain': True,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
                
                mlflow.log_metrics({
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                })
                
                # Log model
                if "xgb" in name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'run_id': mlflow.active_run().info.run_id
                }
                
                print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model = model
                    best_name = name
        
        print(f"  Best model: {best_name} (ROC-AUC: {best_score:.4f})")
        
        return {
            'results': results,
            'best_model': best_model,
            'best_name': best_name,
            'best_score': best_score
        }
    
    def compare_with_production(
        self,
        new_model: Any,
        new_metrics: Dict[str, float],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        min_improvement: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Compare new model with current production model.
        
        Args:
            new_model: Newly trained model
            new_metrics: Metrics from new model
            X_test, y_test: Test data
            min_improvement: Minimum improvement required to deploy
            
        Returns:
            Tuple of (should_deploy, reason)
        """
        print("\nComparing with production model...")
        
        try:
            prod_model_path = self.artifacts_dir / "best_model_final.pkl"
            prod_model = joblib.load(prod_model_path)
            
            # Evaluate production model
            y_pred_prod = prod_model.predict(X_test)
            y_pred_proba_prod = prod_model.predict_proba(X_test)[:, 1]
            
            prod_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_prod),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_prod)
            }
            
            print(f"  Production model ROC-AUC: {prod_metrics['roc_auc']:.4f}")
            print(f"  New model ROC-AUC: {new_metrics['roc_auc']:.4f}")
            
            improvement = new_metrics['roc_auc'] - prod_metrics['roc_auc']
            
            if improvement > min_improvement:
                reason = f"New model improves ROC-AUC by {improvement:.4f}"
                should_deploy = True
            elif improvement > 0:
                reason = f"Improvement ({improvement:.4f}) below threshold ({min_improvement})"
                should_deploy = False
            else:
                reason = f"New model performs worse (decline: {abs(improvement):.4f})"
                should_deploy = False
            
            print(f"  Decision: {'DEPLOY' if should_deploy else 'KEEP PRODUCTION'}")
            print(f"  Reason: {reason}")
            
            return should_deploy, reason
            
        except FileNotFoundError:
            print("  No production model found. Deploying new model.")
            return True, "Initial model deployment"
    
    def deploy_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float]
    ):
        """
        Deploy the new model by DIRECTLY overwriting best_model_final.pkl
        No backups. Clean. Production-ready. Portfolio-perfect.
        """
        print("\nDeploying new model to production...")

        prod_model_path = self.artifacts_dir / "best_model_final.pkl"

        # DIRECTLY OVERWRITE — this is what real companies do in CI/CD
        joblib.dump(model, prod_model_path)
        print(f"   → DEPLOYED: {model_name} is now the live production model")
        print(f"   → Path: {prod_model_path}")
        print(f"   → ROC-AUC: {metrics['roc_auc']:.4f}")

        # Update deployment metadata
        metadata = {
            'deployment_timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'roc_auc': round(metrics['roc_auc'], 4),
            'accuracy': round(metrics['accuracy'], 4),
            'status': 'active',
            'source': 'automated_retraining'
        }

        metadata_path = self.artifacts_dir / "deployment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("   → Deployment metadata updated")
        print("   Deployment complete!")
    
    def run_retraining(
        self,
        trigger_reason: str = "Manual trigger",
        experiment_name: str = "ChurnPrediction-Retraining"
    ) -> Dict[str, Any]:
        """
        Execute complete retraining pipeline with MLflow tracking.
        """
        # === START START-ALERT ===
        alerter = get_alerter()
        alerter.alert_retraining_started(trigger_reason)
        # === END START-ALERT ===
        
        print("=" * 60)
        print("AUTOMATED MODEL RETRAINING")
        print(f"Trigger: {trigger_reason}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        
        with mlflow.start_run(
            run_name=f"retraining_{start_time.strftime('%Y%m%d_%H%M%S')}",
            experiment_id=experiment_id
        ):
            mlflow.log_param('trigger_reason', trigger_reason)
            
            # ---------------------------------------------------------
            # 1. Loading data (UPDATED TO INCLUDE NEW BATCHES)
            # ---------------------------------------------------------
            print("\n1. Loading data...")
            data_path = self.data_dir / "final_processed_data.csv"
            original_df = pd.read_csv(data_path)
            
            # NEW: Look for production data in the monitoring folder
            monitoring_dir = self.project_root / "monitoring"
            new_data_files = list(monitoring_dir.glob("production_batch_*.csv"))
            
            if new_data_files:
                print(f"   Found {len(new_data_files)} new data batches from monitoring.")
                new_dfs = [pd.read_csv(f) for f in new_data_files]
                
                # FIX: Align columns to avoid NaNs
                # We only care about selected_features + Churn. 
                # Dropping extra columns prevents "Excessive missing values" errors.
                selected_features = joblib.load(self.artifacts_dir / "selected_features.pkl")
                required_cols = selected_features + ['Churn']
                
                # Filter original df
                original_clean = original_df[required_cols].copy()
                
                # Filter and clean new batches
                new_dfs_clean = []
                for temp_df in new_dfs:
                    # Ensure all columns exist, fill missing with 0 or drop if critical
                    if set(required_cols).issubset(temp_df.columns):
                        new_dfs_clean.append(temp_df[required_cols])
                
                if new_dfs_clean:
                    df = pd.concat([original_clean] + new_dfs_clean, ignore_index=True)
                    print(f"   Merged data: {len(original_clean)} original + {len(df) - len(original_clean)} new samples.")
                else:
                    df = original_clean
            else:
                print("   No new data found. Using original dataset only.")
                df = original_df

            print(f"   Total Training Samples: {len(df)}")
            mlflow.log_metric('data_samples', len(df))
            
            # ---------------------------------------------------------
            # 2. Validating data
            # ---------------------------------------------------------
            print("\n2. Validating data...")
            is_valid, message = self.validate_data(df)
            mlflow.log_param('data_valid', is_valid)
            
            if not is_valid:
                print(f"   ERROR: {message}")
                summary = {
                    'success': False,
                    'reason': 'Data validation failed',
                    'message': message
                }
                mlflow.log_param('retraining_status', 'failed')
                return summary
            
            print(f"   {message}")
            
            # ---------------------------------------------------------
            # 3. Preparing data
            # ---------------------------------------------------------
            print("\n3. Preparing data...")
            selected_features = joblib.load(self.artifacts_dir / "selected_features.pkl")
            
            # Ensure all columns exist (handle cases where new batch might miss columns)
            # For this simulation, we assume schema is consistent
            X = df[selected_features]
            y = df['Churn']
            
            # Stratify ensures we keep the class balance, which might have changed due to drift
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            print(f"   Training set: {len(X_train)} samples")
            print(f"   Test set: {len(X_test)} samples")
            mlflow.log_metric('train_samples', len(X_train))
            mlflow.log_metric('test_samples', len(X_test))
            
            # ---------------------------------------------------------
            # 4. Training models
            # ---------------------------------------------------------
            print("\n4. Training models...")
            training_results = self.train_models(
                X_train, y_train, X_test, y_test, experiment_id
            )
            
            best_model = training_results['best_model']
            best_name = training_results['best_name']
            best_metrics = training_results['results'][best_name]['metrics']
            
            mlflow.log_param('best_model', best_name)
            mlflow.log_metric('best_roc_auc', best_metrics['roc_auc'])
            
            # ---------------------------------------------------------
            # 5. Comparing with production
            # ---------------------------------------------------------
            print("\n5. Comparing with production...")
            # NOTE: We compare using the NEW test set (which contains drift).
            # The old production model will likely fail here, while the new model
            # (which saw the drift in training) should perform better.
            should_deploy, comparison_reason = self.compare_with_production(
                best_model,
                best_metrics,
                X_test,
                y_test
            )
            
            mlflow.log_param('should_deploy', should_deploy)
            mlflow.log_param('comparison_reason', comparison_reason)
            
            # ---------------------------------------------------------
            # 6. Deploy if better
            # ---------------------------------------------------------
            if should_deploy:
                print("\n6. Deploying model...")
                self.deploy_model(best_model, best_name, best_metrics)
                mlflow.log_param('retraining_status', 'deployed')
            else:
                print("\n6. Keeping production model")
                mlflow.log_param('retraining_status', 'not_deployed')
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'success': True,
                'deployed': should_deploy,
                'trigger_reason': trigger_reason,
                'comparison_reason': comparison_reason,
                'best_model': best_name,
                'best_score': best_metrics['roc_auc'],
                'duration_seconds': duration,
                'timestamp': start_time.isoformat()
            }
            
            mlflow.log_metric('duration_seconds', duration)
            
            # Log summary
            summary_str = json.dumps(summary, indent=2)
            mlflow.log_text(summary_str, "retraining_summary.json")
            # === START COMPLETION-ALERT ===
            alerter.alert_retraining_completed(summary)
            # === END COMPLETION-ALERT ===
        
        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Deployed: {should_deploy}")
        print("=" * 60)
        
        return summary


def main(experiment_id: str):
    """
    Main function to run retraining as part of pipeline.
    
    Args:
        experiment_id: MLflow experiment ID
    """
    pipeline = RetrainingPipeline(PROJECT_ROOT)
    
    summary = pipeline.run_retraining(
        trigger_reason="Scheduled retraining check"
    )
    
    # Save summary
    monitoring_dir = PROJECT_ROOT / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = monitoring_dir / f"retraining_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trigger_retraining.py <experiment_id>")
        sys.exit(1)
    
    main(sys.argv[1])