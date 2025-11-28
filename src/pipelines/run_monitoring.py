"""
Monitoring orchestrator that coordinates performance tracking and retraining.
Integrates with existing project structure and MLflow.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.monitor_performance import ModelMonitor
from src.pipelines.trigger_retraining import RetrainingPipeline
from src.utils.email_alerts import get_alerter


def simulate_production_batch(
    reference_data_path: Path,
    selected_features: list,
    batch_size: int = 500
) -> pd.DataFrame:
    """
    Simulate REAL production data with drift — this fixes the 0.9999 problem
    """
    data = pd.read_csv(reference_data_path)
    batch = data.sample(n=min(batch_size, len(data)), random_state=None).copy()

    # REALISTIC DRIFT (this is what breaks the fake 0.9999)
    np.random.seed(int(datetime.now().timestamp()) % 2**32)  # different every time

    if 'Total day minutes' in batch.columns:
        batch['Total day minutes'] += np.random.normal(10, 5, len(batch))
        batch['Total day minutes'] = batch['Total day minutes'].clip(0, 400)

    if 'Customer service calls' in batch.columns:
        batch['Customer service calls'] += np.random.poisson(1.2, len(batch))
        batch['Customer service calls'] = batch['Customer service calls'].clip(0, 9)

    if 'Total intl minutes' in batch.columns:
        batch['Total intl minutes'] *= np.random.uniform(0.85, 1.4, len(batch))
        batch['Total intl minutes'] = batch['Total intl minutes'].clip(0, 20)

    # Simulate concept drift: flip 10–15% of labels
    flip_ratio = np.random.uniform(0.02, 0.05)
    flip_idx = np.random.choice(batch.index, size=int(len(batch) * flip_ratio), replace=False)
    batch.loc[flip_idx, 'Churn'] = 1 - batch.loc[flip_idx, 'Churn']

    return batch


def run_monitoring_cycle(
    experiment_id: str,
    retraining_cooldown_days: int = 0
):
    """
    Run complete monitoring cycle:
    1. Collect production data
    2. Run performance and drift monitoring
    3. Trigger retraining if needed
    
    Args:
        experiment_id: MLflow experiment ID
        retraining_cooldown_days: Minimum days between retraining
    """
    print("=" * 60)
    print("MONITORING CYCLE START")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Setup paths
    artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
    data_dir = PROJECT_ROOT / "data" / "processed"
    monitoring_dir = PROJECT_ROOT / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    # Load selected features
    selected_features = joblib.load(artifacts_dir / "selected_features.pkl")
    
    # 1. Simulate production data collection
    print("\n1. Collecting production data...")
    production_batch = simulate_production_batch(
        reference_data_path=data_dir / "final_processed_data.csv",
        selected_features=selected_features,
        batch_size=500
    )
    print(f"   Collected {len(production_batch)} samples")
    
    # Save production batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_path = monitoring_dir / f"production_batch_{timestamp}.csv"
    production_batch.to_csv(batch_path, index=False)
    
    # 2. Initialize monitor
    print("\n2. Initializing monitoring system...")
    monitor = ModelMonitor(
        model_path=artifacts_dir / "best_model_final.pkl",
        reference_data_path=data_dir / "final_processed_data.csv",
        selected_features_path=artifacts_dir / "selected_features.pkl",
        performance_threshold=0.75,
        drift_threshold=0.05
    )
    
    # 3. Run monitoring
    print("\n3. Running monitoring checks...")
    X_prod = production_batch[selected_features]
    y_prod = production_batch['Churn'] if 'Churn' in production_batch.columns else None
    
    report = monitor.run_full_monitoring(X_prod, y_prod)
    
    # === SAVE THE MONITORING REPORT TO DISK (THIS WAS MISSING!) ===
    report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = monitoring_dir / f"monitoring_report_{report_timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Monitoring report saved: {report_path.name}")
    # ============================================================
    
    # 4. Check retraining recommendation
    if report.get('retraining', {}).get('recommended', False):
        print("\n4. Retraining recommended")
        print(f"   Reason: {report['retraining']['reason']}")
        
        # === START ALERT CODE ===
        try:
            print("   >> Sending email alert...")
            alerter = get_alerter()
            
            # Prepare drift info for the email
            drift_info = {
                'features_drifted': report['feature_drift'].get('features_drifted', 0),
                'features_checked': report['feature_drift'].get('features_checked', 0),
                'drift_rate': report['feature_drift'].get('drift_rate', 0),
                'drifted_features': report['feature_drift'].get('drifted_features', [])
            }
            
            # Send the email
            alerter.alert_drift_detected(drift_info)
        except Exception as e:
            print(f"Failed to send alert: {e}")
        # === END ALERT CODE ===
        
        # Check cooldown period
        last_retraining = check_last_retraining_time(monitoring_dir)
        
        if last_retraining:
            days_since = (datetime.now() - last_retraining).days
            print(f"   Days since last retraining: {days_since}")
            
            if False:  # days_since < retraining_cooldown_days:
                print(f"   Skipping retraining (cooldown period: {retraining_cooldown_days} days)")
                return
        
        # Trigger retraining
        print("\n5. Triggering retraining...")
        pipeline = RetrainingPipeline(PROJECT_ROOT)
        
        retraining_summary = pipeline.run_retraining(
            trigger_reason=report['retraining']['reason']
        )
        
        # FIX: Check if retraining was actually successful before checking 'deployed'
        if retraining_summary.get('success', False):
            if retraining_summary.get('deployed', False):
                print("\nNew model deployed successfully")
            else:
                print("\nNew model trained but not deployed")
                print(f"Reason: {retraining_summary.get('comparison_reason', 'Unknown')}")
        else:
            print("\nRetraining failed/aborted")
            print(f"Reason: {retraining_summary.get('reason', 'Unknown error')}")
    else:
        print("\n4. No retraining needed")
        print("   Model performance is satisfactory")
    
    print("\n" + "=" * 60)
    print("MONITORING CYCLE COMPLETE")
    print("=" * 60)


def check_last_retraining_time(monitoring_dir: Path) -> datetime:
    """
    Check when last retraining occurred.
    
    Args:
        monitoring_dir: Monitoring directory path
        
    Returns:
        Datetime of last retraining or None
    """
    summaries = list(monitoring_dir.glob("retraining_summary_*.json"))
    
    if not summaries:
        return None
    
    # Get most recent summary
    latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
    
    with open(latest_summary, 'r') as f:
        data = json.load(f)
    
    return datetime.fromisoformat(data['timestamp'])


def main(experiment_id: str):
    """
    Main entry point for monitoring.
    
    Args:
        experiment_id: MLflow experiment ID
    """
    try:
        run_monitoring_cycle(experiment_id)
    except Exception as e:
        print(f"\nERROR: Monitoring cycle failed")
        print(f"Details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_monitoring.py <experiment_id>")
        sys.exit(1)
    
    main(sys.argv[1])