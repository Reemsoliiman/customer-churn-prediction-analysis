# **Monitoring System Structure**

A simple, robust, and fully integrated monitoring system designed to track model performance, detect drift, and trigger automated retraining. This setup blends seamlessly with your existing project structure and MLflow tracking.

## **Files Added to the Project**

```
src/pipelines/
├── monitor_performance.py    # Core monitoring logic
├── trigger_retraining.py     # Automated retraining pipeline
└── run_monitoring.py         # Monitoring orchestrator

monitoring/                   # Auto-created directory
├── monitoring_report_*.json
├── retraining_summary_*.json
└── production_batch_*.csv

MONITORING.md                 # Full documentation
run_all.py    # Enhanced pipeline runner with monitoring
```

---

# **System Architecture**

```
Production Data
      |
      v
[monitor_performance.py]
      |
      +-- Performance Check  → MLflow Logging
      +-- Drift Detection    → MLflow Logging
      +-- Alert Generation   → Console / Logs
      |
      v
   Decision Point
      |
      +-- No Action Needed → Continue Monitoring
      |
      +-- Retraining Needed → [trigger_retraining.py]
                                   |
                                   +-- Train New Models
                                   +-- Compare with Production
                                   +-- Deploy if Better
                                   +-- Log to MLflow
```

---

# **Three Main Components**

---

## **1. Performance Monitoring**

*File: `monitor_performance.py`*

### **What It Does**

* Evaluates the production model on recent data batches
* Detects:

  * **Feature drift** (Kolmogorov–Smirnov test)
  * **Prediction drift** (score distribution changes)
  * **Target drift / concept drift** (Chi-square test)
* Applies thresholds to determine model health
* Recommends retraining when issues are detected

### **Key Metrics Tracked**

* AUC, Accuracy, Precision, Recall, F1
* Drift rate (percentage of drifting features)
* P-values and KS statistics per feature

### **MLflow Integration**

* Logs all metrics and parameters
* Stores monitoring results as artifacts
* Creates separate experiment for monitoring runs

---

## **2. Automated Retraining**

*File: `trigger_retraining.py`*

### **What It Does**

* Validates data quality before retraining
* Trains both Random Forest and XGBoost
* Compares performance against the production model
* Deploys a new model **only when it is meaningfully better** (>1% ROC-AUC improvement)
* Automatically backs up the current production model

### **MLflow Integration**

* Uses the same structure as the original training pipeline
* Nested MLflow runs for each candidate model
* Logs all hyperparameters, metrics, and final decisions
* Stores deployment metadata and summaries

---

## **3. Orchestration Layer**

*File: `run_monitoring.py`*

### **What It Does**

* Simulates or collects a batch of production data
* Runs performance monitoring + drift detection
* Reads retraining recommendations
* Enforces retraining cooldown periods
* Automatically triggers and oversees retraining

### **Cooldown Mechanism**

* Prevents unnecessary retraining
* Ensures at least **7 days** (configurable) between retraining cycles

---

# **How the System Meets Business Requirements**

---

## **Requirement 1: Track Model Performance Over Time**

### **Solution:**

* Monitoring script evaluates each production batch
* Logs all performance metrics to MLflow
* Saves JSON reports in `monitoring/`
* Historical tracking available via MLflow UI

---

## **Requirement 2: Detect Drift**

### **Solution:**

* **Feature Drift** → KS test per feature with p-value thresholds
* **Prediction Drift** → KS test on predicted probabilities
* **Target Drift (Concept Drift)** → Chi-square test on label distribution
* Full drift information logged and visualized in MLflow

---

## **Requirement 3: Establish Alerts**

### **Solution:**

Alerts triggered when:

* ROC-AUC < **0.75**
* Drift p-value < **0.05**
* Feature drift rate > **30%**
* Concept drift detected

Alerts currently appear in:

* Console logs
* Monitoring JSON reports

(Ready for future integration: Slack, Email, PagerDuty)

---

## **Requirement 4: Automated Retraining Plan**

### **Solution:**

* Retraining triggers when:

  * Performance drops below threshold
  * Drift is significant
  * Concept drift detected
* Retraining cooldown prevents retraining fatigue
* New model must outperform production (>1% improvement)
* Full audit trail stored in MLflow

---

# **Quick Start**

---

### **1. Run Initial Training**

```bash
python run_all.py
```

Creates baseline model and reference dataset.

---

### **2. Run Monitoring Manually**

```bash
# Get experiment ID
EXP_ID=$(python -c "import mlflow; client = mlflow.tracking.MlflowClient(); exp = client.get_experiment_by_name('ChurnPrediction-Pipeline'); print(exp.experiment_id)")

# Run monitoring
python src/pipelines/monitor_performance.py $EXP_ID
```

---

### **3. Run Full Monitoring Cycle (with possible retraining)**

```bash
python src/pipelines/run_monitoring.py $EXP_ID
```

This will:

* Collect production batch
* Evaluate performance
* Detect drift
* Recommend/trigger retraining
* Deploy if new model is better

---

### **4. View Results**

```bash
mlflow ui
```

Navigate to:

* **ChurnPrediction-Monitoring**
* **ChurnPrediction-Retraining**

---

# **Configuration & Thresholds**

Default settings (adjustable):

```python
performance_threshold = 0.75      # Minimum acceptable performance
drift_threshold       = 0.05      # p-value threshold for drift
retraining_cooldown_days = 7      # Minimum days between retraining
min_improvement = 0.01            # 1% improvement required for deployment
```

---

# **Monitoring Workflow Summary**

```
1. Production → Collect batch

2. Monitoring → Evaluate performance
              → Detect drift
              → Log results to MLflow

3. Decision → OK → Continue
             → Issue → Trigger retraining

4. Retraining → Validate data
              → Train models
              → Compare with production
              → Deploy best model

5. Deployment → Backup old model
              → Save new model
              → Store metadata

6. Continue → Wait for next cycle
```

---

# **Key Advantages**

* Fully integrated with existing MLflow setup
* Uses familiar project structure
* Lightweight but production-ready
* Automatic backups + safe deployments
* Clear monitoring history & reproducibility
