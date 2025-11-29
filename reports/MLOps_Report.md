# MLOps Report: Experiment Tracking, Deployment, and Retraining

## Executive Summary

This report documents the MLOps infrastructure implemented for the Customer Churn Prediction project. The system includes MLflow experiment tracking, automated model deployment, production monitoring, drift detection, and automated retraining capabilities.

---

## System Architecture

### Pipeline Orchestration

```
Data Collection → Preprocessing → Feature Engineering → Model Training → 
Evaluation → Model Selection → Deployment → Monitoring → Retraining
```

**Orchestrator**: `run_all.py` - Executes all 7 pipeline steps automatically

---

## 1. Experiment Tracking with MLflow

### Implementation

**Tool**: MLflow 2.17.0  
**Tracking URI**: Local file system (`mlruns/` directory)  
**Experiment Name**: `ChurnPrediction-Pipeline`

### What Gets Tracked

#### Metrics
- Model performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Cross-validation scores
- Monitoring metrics (drift rates, performance degradation)
- Retraining comparison metrics

#### Parameters
- Hyperparameters for each model
- Feature engineering parameters
- Model selection criteria
- Monitoring thresholds

#### Artifacts
- Trained models (all formats: sklearn, xgboost)
- Evaluation results (JSON)
- Confusion matrices (HTML)
- Monitoring reports (JSON)
- Production batches (CSV)
- Retraining summaries (JSON)
- Selected features (pickle)

#### Code & Scripts
- Pipeline scripts logged as artifacts
- Standard output and error logs
- Model training code versions

### Nested Runs Structure

```
ChurnPrediction-Pipeline (Experiment)
├── 1/7 Validate data
├── 2/7 Merge raw data
├── 3/7 Pre-processing
├── 4/7 Feature engineering
├── 5/7 Train models
│   ├── logistic_regression (nested)
│   ├── decision_tree (nested)
│   ├── random_forest (nested)
│   └── xgboost (nested)
├── 6/7 Evaluation
└── 7/7 Monitor performance
```

### MLflow Integration Points

**Training Pipeline** (`src/pipelines/train.py`):
- Logs hyperparameters for each model
- Logs cross-validation scores
- Logs test set performance
- Saves models to MLflow model registry

**Evaluation Pipeline** (`src/pipelines/evaluate.py`):
- Logs comprehensive metrics for all models
- Logs best model selection
- Logs confusion matrix visualization
- Logs evaluation results JSON

**Monitoring Pipeline** (`src/pipelines/monitor_performance.py`):
- Logs performance metrics on production data
- Logs drift detection results
- Logs monitoring reports

**Retraining Pipeline** (`src/pipelines/trigger_retraining.py`):
- Logs retraining triggers
- Logs new model performance
- Logs deployment decisions
- Logs comparison with production model

---

## 2. Model Deployment

### Deployment Architecture

#### FastAPI API Service
- **File**: `src/api/main.py`
- **Endpoint**: `POST /predict`
- **Features**:
  - Real-time predictions
  - SHAP explanations (top 6 features)
  - Input validation via Pydantic schemas
  - Error handling and logging

**Deployment Command**:
```bash
uvicorn src.api.main:app --reload --port 8000
```

#### Streamlit Dashboard
- **File**: `src/app/streamlit_app.py`
- **Features**:
  - Interactive prediction interface
  - Monitoring dashboard
  - Analysis dashboard with visualizations
  - Real-time model explanations

**Deployment Command**:
```bash
streamlit run src/app/streamlit_app.py
```

### Model Serving

**Model Loading**:
- Production model: `models/artifacts/best_model_final.pkl`
- Selected features: `models/artifacts/selected_features.pkl`
- Feature engineering: Consistent via `src/utils/helpers.py`

**Inference Pipeline**:
1. Input validation (Pydantic schemas)
2. Feature engineering (same as training)
3. Feature alignment (using selected features)
4. Prediction (model inference)
5. SHAP explanation generation
6. Response formatting

---

## 3. Production Monitoring

### Monitoring System Components

#### Core Monitoring (`src/pipelines/monitor_performance.py`)

**Class**: `ModelMonitor`

**Capabilities**:
1. **Performance Evaluation**
   - Calculates accuracy, precision, recall, F1-score, ROC-AUC
   - Compares against performance threshold (default: 0.75 ROC-AUC)
   - Logs metrics to MLflow

2. **Feature Drift Detection**
   - Method: Kolmogorov-Smirnov (KS) test per feature
   - Threshold: p-value < 0.05 indicates drift
   - Calculates drift rate (% of features with drift)
   - Identifies drifted features

3. **Prediction Drift Detection**
   - Method: KS test on predicted probability distributions
   - Compares production predictions to reference distribution
   - Detects changes in model output patterns

4. **Target Drift Detection (Concept Drift)**
   - Method: Chi-square test on label distributions
   - Detects changes in churn rate over time
   - Indicates concept drift (changing relationships)

### Monitoring Workflow

```
1. Collect Production Batch
   ↓
2. Evaluate Performance
   ↓
3. Detect Feature Drift (KS test)
   ↓
4. Detect Prediction Drift (KS test)
   ↓
5. Detect Target Drift (Chi-square test)
   ↓
6. Generate Monitoring Report
   ↓
7. Log to MLflow
   ↓
8. Recommend Retraining (if needed)
```

### Monitoring Reports

**Location**: `monitoring/monitoring_report_*.json`

**Contents**:
- Timestamp
- Performance metrics
- Feature drift details (drifted features, p-values, KS statistics)
- Prediction drift status
- Target drift status
- Retraining recommendation (boolean + reason)

### Alert System

**Current Implementation**:
- Console logs for real-time alerts
- MLflow tracking for historical analysis
- JSON reports for programmatic access

**Alert Triggers**:
- Performance drop: ROC-AUC < 0.75
- Feature drift: p-value < 0.05 for >30% of features
- Concept drift: Target distribution shift detected
- Prediction drift: Model output distribution changed

**Future Enhancements** (infrastructure ready):
- Email notifications (`src/utils/email_alerts.py`)
- Slack webhook integration
- PagerDuty for critical alerts

---

## 4. Automated Retraining

### Retraining Pipeline (`src/pipelines/trigger_retraining.py`)

**Class**: `RetrainingPipeline`

### Retraining Workflow

```
1. Validate Data Quality
   ↓
2. Train New Models (Random Forest + XGBoost)
   ↓
3. Evaluate New Models
   ↓
4. Compare with Production Model
   ↓
5. Decision: Deploy if Improvement > 1% ROC-AUC
   ↓
6. Backup Current Model
   ↓
7. Deploy New Model (if better)
   ↓
8. Log Deployment Metadata
```

### Retraining Triggers

**Automatic Triggers**:
1. Performance degradation (ROC-AUC < threshold)
2. Significant feature drift (>30% features drifted)
3. Concept drift detected
4. Prediction drift detected

**Manual Triggers**:
- Can be run manually via: `python src/pipelines/trigger_retraining.py <experiment_id>`

### Retraining Criteria

**Deployment Decision**:
- New model must improve ROC-AUC by **>1%** over production model
- Prevents unnecessary deployments from minor fluctuations
- Ensures meaningful improvements

**Model Comparison**:
- Compares Random Forest and XGBoost candidates
- Selects best candidate
- Compares against production model
- Only deploys if significant improvement

### Cooldown Mechanism

**Implementation**: `src/pipelines/run_monitoring.py`

**Purpose**: Prevent retraining thrashing

**Status**: Currently **DISABLED** in code
- Default parameter: `retraining_cooldown_days = 0` (no cooldown by default)
- Cooldown logic exists but is disabled with `if False:` (line 155)
- Can be enabled by changing `if False:` to `if days_since < retraining_cooldown_days:`
- When enabled, would enforce minimum days between retraining cycles

### Retraining Artifacts

**Saved Files**:
- `monitoring/retraining_summary_*.json`: Retraining decision and metrics
- Model backups: Previous production models archived
- MLflow runs: Complete retraining history

---

## 5. Monitoring Orchestration

### Orchestrator (`src/pipelines/run_monitoring.py`)

**Function**: Coordinates monitoring and retraining cycles

**Workflow**:
1. Simulate/collect production data batch
2. Initialize ModelMonitor
3. Run full monitoring cycle
4. Check retraining recommendations
5. Trigger retraining if needed
6. Handle alerts and notifications

**Production Data Simulation**:
- Simulates realistic production batches with drift
- Adds noise to features (day minutes, service calls, intl minutes)
- Simulates concept drift (label flipping 2-5%)
- Saves batches to `monitoring/production_batch_*.csv`

---

## 6. Configuration & Thresholds

### Default Settings

```python
# Performance Monitoring
performance_threshold = 0.75      # Minimum acceptable ROC-AUC

# Drift Detection
drift_threshold = 0.05            # p-value threshold for drift
drift_rate_threshold = 0.30       # 30% features drifted = significant

# Retraining
retraining_cooldown_days = 0      # Default: no cooldown (currently disabled in code)
min_improvement = 0.01            # 1% ROC-AUC improvement required
```

### Configuration Files

- **Project Config**: `config.yaml` (paths, dataset info)
- **Pipeline Config**: Hardcoded in pipeline scripts (can be externalized)

---

## 7. MLflow Experiments

### Experiment Structure

**Main Experiment**: `ChurnPrediction-Pipeline`
- All training and evaluation runs
- Nested runs for each model
- Complete pipeline traceability

**Monitoring Experiment**: (if separate)
- Monitoring cycles
- Performance tracking over time
- Drift detection history

**Retraining Experiment**: (if separate)
- Retraining runs
- Model comparison results
- Deployment decisions

### Run Naming Convention

- `1/7 Validate data`
- `2/7 Merge raw data`
- `3/7 Pre-processing`
- `4/7 Feature engineering`
- `5/7 Train models`
- `6/7 Evaluation`
- `7/7 Monitor performance`
- `logistic_regression` (nested)
- `random_forest` (nested)
- etc.

---

## 8. Model Versioning

### Version Control Strategy

**File-Based Versioning**:
- Models saved with timestamps in filenames
- Previous models backed up before deployment
- Version metadata in MLflow

**MLflow Model Registry**:
- Models logged to MLflow with version tracking
- Can promote models through stages (Staging → Production)
- Model lineage tracking

### Artifact Management

**Stored Artifacts**:
- `models/artifacts/best_model_final.pkl`: Current production model
- `models/trained_models/*.pkl`: All trained models
- `models/artifacts/selected_features.pkl`: Feature list
- `models/artifacts/evaluation_results.json`: Evaluation metrics
- `models/artifacts/test_data.pkl`: Test set for evaluation

---

## 9. Reproducibility

### Reproducibility Features

1. **Random Seeds**: Fixed random_state=42 throughout
2. **Code Logging**: Pipeline scripts logged to MLflow
3. **Parameter Logging**: All hyperparameters tracked
4. **Data Versioning**: Reference datasets stored
5. **Environment Tracking**: Requirements.txt versioned

### MLflow Tracking Benefits

- Complete experiment history
- Easy comparison of runs
- Parameter and metric tracking
- Artifact storage and retrieval
- Model lineage

---

## 10. Deployment Workflow

### Initial Deployment

```
1. Train models (run_all.py)
2. Evaluate and select best model
3. Save best model to artifacts/
4. Deploy via FastAPI/Streamlit
5. Start monitoring cycle
```

### Continuous Deployment

```
1. Monitoring detects issue
2. Retraining triggered
3. New models trained
4. Performance compared
5. If better → Deploy new model
6. Backup old model
7. Update production
```

---

## 11. Monitoring Dashboard

### Streamlit Monitoring Page

**Features**:
- Current model performance metrics
- Drift detection status
- Feature drift rate
- Prediction drift status
- Retraining recommendations
- Historical trends (via MLflow UI)

**Access**: `streamlit run src/app/streamlit_app.py` → Monitoring Dashboard page

---

## 12. Key Advantages

### Production-Ready Features

1. **Automated Monitoring**: Continuous performance tracking
2. **Drift Detection**: Statistical tests for data and concept drift
3. **Automated Retraining**: Self-healing model system
4. **Safe Deployments**: Only deploy if meaningful improvement
5. **Complete Traceability**: MLflow tracks everything
6. **Alert System**: Notifications for issues (extensible)

### MLOps Best Practices

- ✅ Experiment tracking (MLflow)
- ✅ Model versioning
- ✅ Automated testing
- ✅ Monitoring and alerting
- ✅ Automated retraining
- ✅ Reproducibility
- ✅ Model serving (API + Dashboard)

---

## 13. Future Enhancements

### Planned Improvements

1. **Cloud Deployment**: AWS/GCP/Azure configurations
2. **Enhanced Alerts**: Email, Slack, PagerDuty integration
3. **A/B Testing**: Framework for model comparison
4. **Feature Store**: Centralized feature management
5. **Model Registry**: Advanced model lifecycle management
6. **CI/CD Pipeline**: Automated testing and deployment
7. **Grafana Dashboards**: Real-time monitoring visualization

---

## 14. Usage Examples

### Run Full Pipeline
```bash
python run_all.py
```

### Run Monitoring Cycle
```bash
python src/pipelines/run_monitoring.py <experiment_id>
```

### View MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
```

### Deploy API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Deploy Dashboard
```bash
streamlit run src/app/streamlit_app.py
```

---

**Report Generated From**: `MONITORING.md` and pipeline code analysis  
**MLOps Infrastructure**: MLflow, FastAPI, Streamlit, Automated Monitoring  
**Status**: Production-ready with continuous monitoring and retraining

