# Telco Customer Churn Prediction + Production Monitoring
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Monitoring](https://img.shields.io/badge/Monitoring-Drift%20%26%20Retraining-brightgreen.svg)

A **production-grade MLOps system** for predicting telecom customer churn — with automated training, model selection, serving, explainability, and continuous performance monitoring with drift detection and automated retraining triggers.

## Project Overview & Key Features

- End-to-end ML pipeline orchestrated by `run_all.py`
- MLflow experiment tracking for every step
- Feature engineering + RFE (selects 20 best features)
- Model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Best model automatically selected and saved
- SHAP explanations for every prediction
- FastAPI prediction endpoint
- Interactive Streamlit dashboard with 3 pages
- Production monitoring with data drift & performance degradation detection
- Automated retraining pipeline (triggers when needed)

---

## Project Structure

```
customer-churn-prediction-analysis/
├── data/
│   ├── raw/                  # churn-bigml-20.csv, churn-bigml-80.csv, merged_churn_data.csv
│   └── processed/            # cleaned_data.csv, final_processed_data.csv
├── models/
│   ├── trained_models/       # All trained models (.pkl files)
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── best_churn_model.pkl
│   └── artifacts/            # Model artifacts and metadata
│       ├── best_model_final.pkl
│       ├── selected_features.pkl
│       ├── test_data.pkl
│       ├── deployment_metadata.json
│       ├── evaluation_results.json
│       └── training_summary.json
├── monitoring/               # Auto-generated reports & production batches
│   ├── monitoring_report_*.json
│   └── production_batch_*.csv
├── notebooks/                # EDA and analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_deployment.ipynb
├── reports/                  # Project documentation and reports
│   ├── Data_Analysis_Report.md
│   ├── EDA_Report.md
│   ├── Feature_Engineering_Summary.md
│   ├── Final_Project_Report.md
│   ├── MLOps_Report.md
│   ├── Model_Evaluation_Report.md
│   └── MONITORING.md         # Monitoring system documentation
├── src/
│   ├── pipelines/            # All pipeline steps + monitoring + retraining
│   │   ├── validate_data.py
│   │   ├── merge_data.py
│   │   ├── preprocess.py
│   │   ├── engineer_features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── monitor_performance.py
│   │   ├── run_monitoring.py
│   │   └── trigger_retraining.py
│   ├── api/                  # FastAPI service
│   │   ├── main.py
│   │   ├── predict.py
│   │   └── schemas.py
│   ├── app/                  # Streamlit dashboard
│   │   └── streamlit_app.py
│   └── utils/                # Shared utilities
│       ├── helpers.py        # Feature engineering functions
│       └── email_alerts.py   # Alert infrastructure
├── visualizations/           # Saved visualization files
│   ├── interactive/          # HTML visualizations (confusion matrices, ROC curves, SHAP, etc.)
│   └── static/               # PNG visualizations (distributions, heatmaps, etc.)
├── mlruns/                   # MLflow tracking database (auto-generated)
├── run_all.py                # Full pipeline orchestrator
├── requirements.txt          # Python dependencies
└── config.yaml               # Project configuration
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Reemsoliiman/customer-churn-prediction-analysis.git
cd customer-churn-prediction-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the two Kaggle dataset files in `data/raw/`:
- `churn-bigml-20.csv`
- `churn-bigml-80.csv`

Download from: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets

---

## Usage

### Run the Full Pipeline (Training + Monitoring)

```bash
python run_all.py
```

This executes **7 steps** automatically:
1. Validate data
2. Merge datasets
3. Preprocess (handle missing values, encode categoricals)
4. Feature engineering + RFE (selects 20 features)
5. Train 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
6. Evaluate & select best model
7. Run production monitoring cycle (drift detection + retraining decision)

### View Results in MLflow UI

```bash
mlflow ui
```

Open → http://localhost:5000

You'll see every run with metrics, parameters, and artifacts including:
- `monitoring_report_*.json` – full drift & performance report
- `production_batch_*.csv` – simulated live data
- `retraining_summary_*.json` – only if retraining was triggered

---

## Model Serving & Interaction

### 1. FastAPI Prediction API

```bash
uvicorn src.api.main:app --reload --port 8000
```

**Endpoint**: `POST http://localhost:8000/predict`

**Request Body** (JSON):
```json
{
  "Account length": 100,
  "International plan": "Yes",
  "Voice mail plan": "No",
  "Number vmail messages": 0,
  "Total day minutes": 180.0,
  "Total eve minutes": 150.0,
  "Total night minutes": 150.0,
  "Total intl minutes": 10.0,
  "Total day calls": 100,
  "Total eve calls": 90,
  "Total night calls": 90,
  "Total intl calls": 3,
  "Customer service calls": 1
}
```

**Response**: Returns churn probability + top 6 SHAP feature explanations.

### 2. Interactive Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Open → http://localhost:8501

**Dashboard Pages**:
- **Predict Churn**: Real-time predictions with interactive sliders + SHAP bar chart
- **Monitoring Dashboard**: Performance tracking, drift detection status, retraining recommendations
- **Analysis Dashboard**: Interactive EDA with filters, visualizations, and insights

---

## Continuous Monitoring & Retraining

### Run Monitoring Cycle Manually

```bash
# First, get your experiment ID from MLflow UI or run_all.py output
# Then run:
python src/pipelines/run_monitoring.py <experiment_id>
```

The monitoring system automatically:
- Samples recent production data (simulated)
- Checks model performance degradation (ROC-AUC, accuracy, precision, recall, F1)
- Detects feature drift (Kolmogorov-Smirnov test per feature)
- Detects prediction drift (distribution changes)
- Detects target drift (if labels available, chi-square test)
- Recommends retraining when performance drops or drift is significant

If retraining is recommended → the system automatically triggers retraining pipeline.

**Note**: Retraining cooldown logic exists in code but is currently disabled (can be enabled in `src/pipelines/run_monitoring.py`).

---

## Model Performance

Typical performance metrics (may vary based on data and hyperparameters):

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.80    | ~0.81   |
| Decision Tree       | ~0.87    | ~0.83   |
| Random Forest       | ~0.94    | ~0.92   |
| **XGBoost**         | **~0.94** | **~0.92** |

*Best model (typically Random Forest or XGBoost, selected by highest ROC-AUC) is automatically saved as `models/artifacts/best_model_final.pkl`*

---

## Tech Stack

**Core ML & Data Science**:
- Python 3.9+
- pandas 2.2.3
- numpy 2.1.2
- scikit-learn 1.5.2
- xgboost 2.1.1
- imbalanced-learn 0.11.0+

**MLOps & Experiment Tracking**:
- MLflow 2.17.0

**Model Explainability**:
- SHAP 0.46.0

**API & Web Framework**:
- FastAPI 0.115.0
- uvicorn 0.30.6
- pydantic 2.9.2

**Visualization & Dashboard**:
- Streamlit 1.39.0
- Plotly 5.24.1

**Utilities**:
- joblib 1.4.2
- requests 2.32.3
- scipy (dependency of scikit-learn, used for statistical tests in monitoring)

---

## Key Technical Highlights

- **Consistent feature engineering**: `src/utils/helpers.py` → used in both training and inference
- **SHAP TreeExplainer**: Fast, accurate explanations for tree-based models (with KernelExplainer fallback)
- **RFE with RandomForest**: Robust feature selection (selects top 20 features)
- **MLflow nested runs**: Complete traceability of all pipeline steps
- **Production-ready monitoring**: Statistical drift tests (KS test, chi-square test)
- **Automated retraining**: Compares new models with production, deploys only if improvement > 1% ROC-AUC
- **Class balancing**: SMOTE oversampling applied during training
- **Hyperparameter tuning**: RandomizedSearchCV for Logistic Regression, Random Forest, and XGBoost

---

## Dataset

**Kaggle Telecom Customer Churn Dataset**  
Source: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets

The project uses the merged dataset (churn-bigml-20.csv + churn-bigml-80.csv) containing customer demographics, usage patterns, and subscription details.

---

## Documentation

- **Monitoring System**: See `reports/MONITORING.md` for detailed documentation on the monitoring and retraining pipeline
- **Project Reports**: See `reports/` directory for comprehensive analysis reports:
  - `MONITORING.md` - Monitoring system documentation
  - `MLOps_Report.md` - MLOps infrastructure and deployment
  - `Final_Project_Report.md` - Complete project overview
  - `Feature_Engineering_Summary.md` - Feature engineering details
  - `Model_Evaluation_Report.md` - Model performance analysis
  - `EDA_Report.md` - Exploratory data analysis
  - `Data_Analysis_Report.md` - Data analysis insights
- **Notebooks**: Explore `notebooks/` directory for detailed EDA and analysis
