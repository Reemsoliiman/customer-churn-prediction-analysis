"""
Monitoring orchestrator that coordinates performance tracking and retraining.
Integrates with existing project structure and MLflow.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.monitor_performance import ModelMonitor
from src.pipelines.trigger_retraining import RetrainingPipeline


def simulate_production_batch(
    reference_data_path: Path,
    selected_features: list,
    batch_size: int = 500
) -> pd.DataFrame:
    """
    Simulate production data batch.
    In production, replace with actual data collection from your system.
    
    Args:
        reference_data_path: Path to reference data
        selected_features: List of feature names
        batch_size: Number of samples to collect
        
    Returns:
        DataFrame with production batch
    """
    # Load reference data
    data = pd.read_csv(reference_data_path)
    
    # Sample random batch
    batch = data.sample(n=min(batch_size, len(data)), random_state=None)
    
    return batch


def run_monitoring_cycle(
    experiment_id: str,
    retraining_cooldown_days: int = 7
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
    
    # 4. Check retraining recommendation
    if report.get('retraining', {}).get('recommended', False):
        print("\n4. Retraining recommended")
        print(f"   Reason: {report['retraining']['reason']}")
        
        # Check cooldown period
        last_retraining = check_last_retraining_time(monitoring_dir)
        
        if last_retraining:
            days_since = (datetime.now() - last_retraining).days
            print(f"   Days since last retraining: {days_since}")
            
            if days_since < retraining_cooldown_days:
                print(f"   Skipping retraining (cooldown period: {retraining_cooldown_days} days)")
                return
        
        # Trigger retraining
        print("\n5. Triggering retraining...")
        pipeline = RetrainingPipeline(PROJECT_ROOT)
        
        retraining_summary = pipeline.run_retraining(
            trigger_reason=report['retraining']['reason']
        )
        
        if retraining_summary['deployed']:
            print("\nNew model deployed successfully")
        else:
            print("\nNew model trained but not deployed")
            print(f"Reason: {retraining_summary['comparison_reason']}")
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