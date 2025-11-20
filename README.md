Here’s your **brand-new, clean, up-to-date README** — with the **monitoring & automated retraining** proudly featured, **tests and Docker removed**, and everything else polished and accurate as of your final working version.

```markdown
# Telco Customer Churn Prediction + Production Monitoring
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Monitoring](https://img.shields.io/badge/Monitoring-Drift%20%26%20Retraining-brightgreen.svg)

A **complete production-grade MLOps system** for predicting telecom customer churn — with automated training, model selection, serving, explainability, **and continuous performance monitoring with drift detection + automated retraining triggers**.

## Project Overview & Key Features

- End-to-end ML pipeline orchestrated by `run_all.py`
- MLflow experiment tracking for every step
- Feature engineering + RFE (20 best features)
- Model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Best model automatically selected and deployed
- SHAP explanations for every prediction
- FastAPI prediction endpoint
- Beautiful interactive Streamlit dashboard
- **Production monitoring with data drift & performance degradation detection**
- **Automated retraining recommendation (with cooldown)**

---

## Project Structure (simplified)

```
customer-churn-prediction-analysis/
├── data/
│   ├── raw/                  # churn-bigml-20.csv + churn-bigml-80.csv
│   └── processed/            # cleaned → final data
├── models/
│   ├── trained_models/       # All trained models
│   └── artifacts/            # best_model_final.pkl, selected_features.pkl, etc.
├── monitoring/               # Auto-generated reports & production batches
├── src/
│   ├── pipelines/            # All pipeline steps + monitoring + retraining
│   ├── api/                  # FastAPI service
│   ├── app/                  # Streamlit dashboard
│   └── utils/helpers.py      # Shared feature engineering
├── mlruns/                   # MLflow tracking database
├── run_all.py                # Full pipeline + monitoring in one command
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Reemsoliiman/customer-churn-prediction-analysis.git
cd customer-churn-prediction-analysis

python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Place the two Kaggle files in `data/raw/`:
- `churn-bigml-20.csv`
- `churn-bigml-80.csv`

### Run the Full Pipeline (Training + Monitoring)

```bash
python run_all.py
```

This executes **all 7 steps** automatically:
1. Validate data
2. Merge datasets
3. Preprocess
4. Feature engineering + RFE
5. Train 4 models
6. Evaluate & select best model
7. **Run production monitoring cycle** (drift detection + retraining decision)

### View Results in MLflow UI

```bash
mlflow ui
```

Open → http://localhost:5000

You’ll see every run with metrics, parameters, and artifacts including:
- `monitoring_report_*.json` – full drift & performance report
- `production_batch_*.csv` – simulated live data
- `retraining_summary_*.json` – only if retraining was triggered

---

## Model Serving & Interaction

### 1. FastAPI Prediction API

```bash
uvicorn src.api.main:app --reload --port 8000
```

→ `POST http://localhost:8000/predict`  
Returns churn probability + top 6 SHAP explanations.

### 2. Interactive Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

→ http://localhost:8501  
Real-time sliders → instant prediction + beautiful SHAP bar chart.

---

## Continuous Monitoring & Retraining

Your system now **watches itself in production**:

```bash
# Run a monitoring cycle anytime (e.g. daily via scheduler)
python -m src.pipelines.run_monitoring <experiment_id>
```

It automatically:
- Samples recent production data
- Checks model performance degradation
- Detects feature & prediction drift (Kolmogorov-Smirnov test)
- Detects target drift (if labels available)
- Recommends retraining only when truly needed
- Respects a 7-day cooldown to avoid thrashing

If performance drops or drift is severe → retraining is triggered automatically.

---

## Model Performance (typical)

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.86    | ~0.84   |
| Decision Tree       | ~0.91    | ~0.87   |
| Random Forest       | ~0.95    | ~0.93   |
| **XGBoost**         | **0.96** | **0.98+** |

*Best model (usually XGBoost) is automatically saved as `best_model_final.pkl`*

---

## Key Technical Highlights

- Consistent feature engineering in `utils/helpers.py` → used in training **and** inference
- SHAP TreeExplainer for fast, accurate explanations
- RFE with RandomForest for robust feature selection
- MLflow nested runs for perfect traceability
- Production-ready monitoring with statistical drift tests
- Zero-downtime retraining logic

---

## Dataset

Kaggle Telecom Customer Churn (merged 20% + 80% datasets)  
https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
