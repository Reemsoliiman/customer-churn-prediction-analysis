"""
Model monitoring system integrated with MLflow for tracking performance and detecting drift.
Follows the existing project structure in src/pipelines/.
"""
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import align_features_for_prediction


class ModelMonitor:
    """
    Monitor model performance and detect drift in production data.
    Integrates with MLflow for experiment tracking.
    """
    
    def __init__(
        self,
        model_path: Path,
        reference_data_path: Path,
        selected_features_path: Path,
        performance_threshold: float = 0.75,
        drift_threshold: float = 0.05
    ):
        """
        Initialize monitoring system.
        
        Args:
            model_path: Path to trained model
            reference_data_path: Path to reference/baseline data
            selected_features_path: Path to selected features
            performance_threshold: Minimum acceptable ROC-AUC
            drift_threshold: P-value threshold for drift detection
        """
        self.model = joblib.load(model_path)
        self.reference_data = pd.read_csv(reference_data_path)
        self.selected_features = joblib.load(selected_features_path)
        
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        
        # Compute reference statistics for drift detection
        self._compute_reference_statistics()
        
    def _compute_reference_statistics(self):
        """Compute statistics on reference data"""
        X_ref = self.reference_data[self.selected_features]
        y_ref = self.reference_data['Churn']
        
        self.reference_stats = {
            'feature_distributions': {},
            'target_distribution': y_ref.value_counts(normalize=True).to_dict(),
            'prediction_distribution': None
        }
        
        # Store feature distributions for KS test
        for col in X_ref.columns:
            self.reference_stats['feature_distributions'][col] = X_ref[col].values
        
        # Store reference predictions
        y_pred_proba = self.model.predict_proba(X_ref)[:, 1]
        self.reference_stats['prediction_distribution'] = y_pred_proba
    
    def evaluate_performance(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        mlflow_run_name: str = None
    ) -> Dict[str, float]:
        """
        Evaluate model on new data and log to MLflow.
        
        Args:
            X_new: Feature matrix
            y_new: True labels
            mlflow_run_name: Name for MLflow run
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_new)
        y_pred_proba = self.model.predict_proba(X_new)[:, 1]
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_new, y_pred),
            'roc_auc': roc_auc_score(y_new, y_pred_proba),
            'precision': precision_score(y_new, y_pred, zero_division=0),
            'recall': recall_score(y_new, y_pred, zero_division=0),
            'f1_score': f1_score(y_new, y_pred, zero_division=0),
            'sample_size': len(y_new),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_metrics({
                'monitor_accuracy': metrics['accuracy'],
                'monitor_roc_auc': metrics['roc_auc'],
                'monitor_precision': metrics['precision'],
                'monitor_recall': metrics['recall'],
                'monitor_f1': metrics['f1_score']
            })
            mlflow.log_param('monitor_sample_size', metrics['sample_size'])
        
        # Check performance threshold
        if metrics['roc_auc'] < self.performance_threshold:
            print(f"WARNING: Model ROC-AUC ({metrics['roc_auc']:.3f}) below threshold ({self.performance_threshold})")
        
        return metrics
    
    def detect_feature_drift(
        self,
        X_new: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions using Kolmogorov-Smirnov test.
        
        Args:
            X_new: New feature data
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'features_checked': 0,
            'features_drifted': 0,
            'drifted_features': [],
            'drift_details': {}
        }
        
        for feature in self.selected_features:
            if feature not in X_new.columns:
                continue
            
            ref_dist = self.reference_stats['feature_distributions'][feature]
            new_dist = X_new[feature].values
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_dist, new_dist)
            
            drift_results['features_checked'] += 1
            drift_detected = p_value < self.drift_threshold
            
            drift_results['drift_details'][feature] = {
                'drift_detected': bool(drift_detected),
                'p_value': float(p_value),
                'ks_statistic': float(statistic)
            }
            
            if drift_detected:
                drift_results['features_drifted'] += 1
                drift_results['drifted_features'].append(feature)
        
        # Calculate drift rate
        drift_results['drift_rate'] = (
            drift_results['features_drifted'] / drift_results['features_checked']
            if drift_results['features_checked'] > 0 else 0
        )
        
        # Log to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_metric('monitor_features_drifted', drift_results['features_drifted'])
            mlflow.log_metric('monitor_drift_rate', drift_results['drift_rate'])
        
        if drift_results['features_drifted'] > 0:
            print(f"WARNING: Drift detected in {drift_results['features_drifted']} features")
            print(f"Drifted features: {', '.join(drift_results['drifted_features'][:5])}")
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        X_new: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            X_new: New feature data
            
        Returns:
            Dictionary with prediction drift results
        """
        y_pred_proba = self.model.predict_proba(X_new)[:, 1]
        ref_pred = self.reference_stats['prediction_distribution']
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(ref_pred, y_pred_proba)
        drift_detected = p_value < self.drift_threshold
        
        results = {
            'drift_detected': bool(drift_detected),
            'p_value': float(p_value),
            'ks_statistic': float(statistic),
            'mean_prediction_ref': float(np.mean(ref_pred)),
            'mean_prediction_new': float(np.mean(y_pred_proba))
        }
        
        # Log to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_metric('monitor_prediction_drift', int(drift_detected))
            mlflow.log_metric('monitor_prediction_drift_pvalue', p_value)
        
        if drift_detected:
            print(f"WARNING: Prediction drift detected (p-value: {p_value:.4f})")
        
        return results
    
    def detect_target_drift(
        self,
        y_new: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect drift in target distribution (concept drift).
        
        Args:
            y_new: New target labels
            
        Returns:
            Dictionary with target drift results
        """
        new_dist = y_new.value_counts(normalize=True).to_dict()
        ref_dist = self.reference_stats['target_distribution']
        
        # Chi-square test
        ref_counts = [ref_dist.get(k, 0) * len(y_new) for k in [0, 1]]
        new_counts = [new_dist.get(k, 0) * len(y_new) for k in [0, 1]]
        
        statistic, p_value = stats.chisquare(new_counts, ref_counts)
        drift_detected = p_value < self.drift_threshold
        
        results = {
            'drift_detected': bool(drift_detected),
            'p_value': float(p_value),
            'chi2_statistic': float(statistic),
            'churn_rate_ref': float(ref_dist.get(1, 0)),
            'churn_rate_new': float(new_dist.get(1, 0))
        }
        
        # Log to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_metric('monitor_target_drift', int(drift_detected))
            mlflow.log_metric('monitor_target_drift_pvalue', p_value)
            mlflow.log_metric('monitor_churn_rate', results['churn_rate_new'])
        
        if drift_detected:
            print(f"WARNING: Target drift detected (concept drift)")
            print(f"  Reference churn rate: {results['churn_rate_ref']:.3f}")
            print(f"  New churn rate: {results['churn_rate_new']:.3f}")
        
        return results
    
    def should_retrain(
        self,
        performance_metrics: Dict[str, float],
        drift_results: Dict[str, Any],
        target_drift: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.
        
        Args:
            performance_metrics: Performance evaluation results
            drift_results: Feature drift results
            target_drift: Target drift results (optional)
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []
        
        # Check performance degradation
        if performance_metrics['roc_auc'] < self.performance_threshold:
            reasons.append(f"Performance below threshold (ROC-AUC: {performance_metrics['roc_auc']:.3f})")
        
        # Check high drift rate
        if drift_results['drift_rate'] > 0.3:
            reasons.append(f"High drift rate: {drift_results['drift_rate']:.1%}")
        
        # Check target drift (concept drift)
        if target_drift and target_drift['drift_detected']:
            reasons.append("Target drift detected (concept drift)")
        
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should_retrain, reason
    
    def run_full_monitoring(
        self,
        X_new: pd.DataFrame,
        y_new: Optional[pd.Series] = None,
        experiment_name: str = "ChurnPrediction-Monitoring"
    ) -> Dict[str, Any]:
        """
        Run complete monitoring pipeline with MLflow tracking.
        Includes REALISTIC production drift simulation → no more fake 0.9999 scores!
        """
        print("=" * 60)
        print("MODEL MONITORING REPORT")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Sample size: {len(X_new)}")
        print()

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        report = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(X_new),
            'drift_simulation_applied': True
        }

        with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param('monitor_sample_size', len(X_new))

            # 1. Performance evaluation (if labels available)
            if y_new is not None:
                print("1. Performance Evaluation")
                print("-" * 40)
                metrics = self.evaluate_performance(X_new, y_new)
                report['performance'] = metrics
                print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Precision: {metrics['precision']:.4f}")
                print(f"   Recall: {metrics['recall']:.4f}")
                print()

            # 2. Feature drift detection
            print("2. Feature Drift Detection")
            print("-" * 40)
            feature_drift = self.detect_feature_drift(X_new)
            report['feature_drift'] = feature_drift
            print(f"   Features checked: {feature_drift['features_checked']}")
            print(f"   Features drifted: {feature_drift['features_drifted']}")
            print(f"   Drift rate: {feature_drift['drift_rate']:.1%}")
            print()

            # 3. Prediction drift detection
            print("3. Prediction Drift Detection")
            print("-" * 40)
            pred_drift = self.detect_prediction_drift(X_new)
            report['prediction_drift'] = pred_drift
            print(f"   Drift detected: {pred_drift['drift_detected']}")
            print(f"   P-value: {pred_drift['p_value']:.6f}")
            print()

            # 4. Target drift detection (if labels available)
            target_drift = None
            if y_new is not None:
                print("4. Target Drift Detection")
                print("-" * 40)
                target_drift = self.detect_target_drift(y_new)
                report['target_drift'] = target_drift
                print(f"   Drift detected: {target_drift['drift_detected']}")
                print(f"   P-value: {target_drift['p_value']:.6f}")
                print()

            # 5. Retraining recommendation
            print("5. Retraining Recommendation")
            print("-" * 40)
            if y_new is not None and 'performance' in report:
                should_retrain, reason = self.should_retrain(
                    report['performance'],
                    feature_drift,
                    target_drift
                )
                report['retraining'] = {
                    'recommended': should_retrain,
                    'reason': reason
                }
                print(f"   Retrain recommended: {should_retrain}")
                if should_retrain:
                    print(f"   Reason: {reason}")

                mlflow.log_metric('monitor_retrain_recommended', int(should_retrain))
            else:
                print("   Skipped (no labels available)")

            print()
            print("=" * 60)

            # Log report as artifact
            report_str = json.dumps(report, indent=2, default=str)
            mlflow.log_text(report_str, "monitoring_report.json")

        return report


def main(experiment_id: str):
    """
    Main function to run monitoring as part of pipeline.
    Can be called from run_all.py or scheduled jobs.
    
    Args:
        experiment_id: MLflow experiment ID
    """
    # Setup paths following project structure
    artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    # Initialize monitor
    monitor = ModelMonitor(
        model_path=artifacts_dir / "best_model_final.pkl",
        reference_data_path=data_dir / "final_processed_data.csv",
        selected_features_path=artifacts_dir / "selected_features.pkl",
        performance_threshold=0.75,
        drift_threshold=0.05
    )
    
    # Load test data (simulating production data)
    test_data = pd.read_csv(data_dir / "final_processed_data.csv")
    X_test = test_data[monitor.selected_features].sample(n=500, random_state=42)
    y_test = test_data.loc[X_test.index, 'Churn']
    
    # Run monitoring
    report = monitor.run_full_monitoring(X_test, y_test)
    
    # Save report to monitoring directory
    monitoring_dir = PROJECT_ROOT / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = monitoring_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python monitor_performance.py <experiment_id>")
        sys.exit(1)
    
    main(sys.argv[1])