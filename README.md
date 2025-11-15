# 📞 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

A production-ready machine learning system for predicting customer churn in telecommunications. This project implements an end-to-end ML pipeline with experiment tracking, model serving, and interactive visualization.

## 🎯 Project Overview

**Goal**: Predict which customers are likely to churn (leave the service) and explain why using SHAP values.

**Key Features**:
- ✅ Automated end-to-end ML pipeline with MLflow tracking
- ✅ Feature engineering with RFE-based selection (20 best features)
- ✅ Multiple model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- ✅ Model interpretability using SHAP explanations
- ✅ FastAPI REST API for predictions
- ✅ Interactive Streamlit dashboard
- ✅ Docker deployment ready

---

## 📁 Project Structure

```
churn-prediction-project/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── churn-bigml-20.csv
│   │   ├── churn-bigml-80.csv
│   │   └── merged_churn_data.csv
│   └── processed/                    # Cleaned and engineered data
│       ├── cleaned_data.csv
│       └── final_processed_data.csv
│
├── models/
│   ├── trained_models/               # Serialized model files
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── random_forest.pkl
│   │   └── xgboost.pkl
│   └── artifacts/                    # Model metadata
│       ├── selected_features.pkl
│       ├── test_data.pkl
│       ├── best_model_final.pkl
│       └── evaluation_results.json
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_deployment_insights.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── utils/                        # Helper functions
│   │   ├── __init__.py
│   │   └── helpers.py                # Feature engineering, preprocessing
│   │
│   ├── pipelines/                    # MLflow pipeline steps
│   │   ├── __init__.py
│   │   ├── validate_data.py          # Step 1: Validate raw data
│   │   ├── merge_data.py             # Step 2: Merge datasets
│   │   ├── preprocess.py             # Step 3: Clean and encode
│   │   ├── engineer_features.py      # Step 4: Feature engineering + RFE
│   │   ├── train.py                  # Step 5: Train models
│   │   └── evaluate.py               # Step 6: Select best model
│   │
│   ├── api/                          # FastAPI service
│   │   ├── __init__.py
│   │   ├── main.py                   # API endpoints
│   │   ├── predict.py                # Prediction logic + SHAP
│   │   └── schemas.py                # Pydantic models
│   │
│   └── app/                          # Streamlit UI
│       ├── __init__.py
│       └── streamlit_app.py
│
├── tests/                            # Unit and integration tests
│   ├── unit/
│   └── integration/
│
├── mlruns/                           # MLflow tracking (auto-created)
│
├── run_all.py                        # Pipeline orchestrator
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container image
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- (Optional) Docker

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Reemsoliiman/customer-churn-prediction-analysis.git
   cd customer-churn-prediction-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   
   Place the Kaggle "Telco Churn" dataset files in `data/raw/`:
   - `churn-bigml-20.csv`
   - `churn-bigml-80.csv`

---

## 📊 Pipeline Execution

### Run Full Pipeline

Execute all 6 steps automatically:

```bash
python run_all.py
```

**Pipeline Steps**:

1. **Validate Data** – Check raw files exist
2. **Merge Data** – Combine 20% + 80% datasets
3. **Preprocess** – Handle missing values, encode categoricals
4. **Feature Engineering** – Create features, RFE selection (20 features)
5. **Train Models** – Train 4 models (LogReg, DecisionTree, RandomForest, XGBoost)
6. **Evaluate** – Select best model by ROC-AUC

### View Results in MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 to see:
- Experiment tracking
- Metrics comparison (accuracy, ROC-AUC)
- Model artifacts

---

## 🔮 Model Serving

### 1. FastAPI REST API

Start the API server:

```bash
cd src
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints**:
- `GET /` – Health check
- `POST /predict` – Churn prediction with SHAP explanation

**Example Request**:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Account length": 100,
    "International plan": "No",
    "Voice mail plan": "Yes",
    "Number vmail messages": 25,
    "Total day minutes": 200.0,
    "Total eve minutes": 180.0,
    "Total night minutes": 150.0,
    "Total intl minutes": 10.0,
    "Total day calls": 90,
    "Total eve calls": 85,
    "Total night calls": 80,
    "Total intl calls": 5,
    "Customer service calls": 2
  }'
```

**Response**:

```json
{
  "churn_probability": 0.23,
  "churn_prediction": false,
  "base_value": 0.14,
  "top_shap_features": [
    {"feature": "Customer service calls", "shap_value": 0.15, "abs_shap": 0.15},
    {"feature": "Total_Minutes", "shap_value": -0.08, "abs_shap": 0.08},
    ...
  ]
}
```

### 2. Streamlit Dashboard

Launch the interactive UI:

```bash
streamlit run src/app/streamlit_app.py
```

Open http://localhost:8501

**Features**:
- Interactive sliders for customer attributes
- Real-time churn prediction
- Visual SHAP explanations
- Top 6 feature impacts highlighted

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/unit -v
```

Run integration tests:

```bash
pytest tests/integration -v
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t churn-prediction:latest .
```

### Run API Container

```bash
docker run -p 8000:8000 churn-prediction:latest
```

### Run Streamlit Container

```bash
docker run -p 8501:8501 \
  -e API_URL=http://api-container:8000/predict \
  churn-prediction:latest \
  streamlit run src/app/streamlit_app.py
```

---

## 📈 Model Performance

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 0.86     | 0.84    |
| Decision Tree       | 0.91     | 0.87    |
| **Random Forest**   | **0.95** | **0.93**|
| XGBoost             | 0.94     | 0.92    |

*Best model selected automatically based on ROC-AUC score*

---

## 🔧 Key Features

### Feature Engineering

- **Customer tenure** (months from account length)
- **Total aggregates** (total minutes/calls across day/eve/night/intl)
- **Usage patterns** (avg daily usage, call frequency, intl usage rate)
- **Behavioral flags** (high service calls, has voicemail)
- **Log transforms** (for skewed distributions)

### Model Interpretability

Uses SHAP (SHapley Additive exPlanations) to provide:
- Feature importance rankings
- Individual prediction explanations
- Force plots showing contribution of each feature

---

## 📝 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  test_size: 0.2

features:
  n_features_to_select: 20
  rfe_estimator: "random_forest"

models:
  random_forest:
    n_estimators: 300
    random_state: 42
  xgboost:
    n_estimators: 300
    eval_metric: "logloss"

mlflow:
  experiment_name: "ChurnPrediction-Pipeline"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
- SHAP library for model interpretability
- MLflow for experiment tracking
- FastAPI and Streamlit for deployment

---

**Built with ❤️ using Python, MLflow, FastAPI, and Streamlit**