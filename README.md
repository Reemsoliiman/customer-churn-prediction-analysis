# Customer Churn Prediction Pipeline

**End-to-end ML system** to predict telecom customer churn using the [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

Includes data validation, preprocessing, feature engineering, model training, evaluation, selection, and two Streamlit apps:
- **Analysis Dashboard** – Explore model insights
- **Live Prediction App** – Predict churn risk with SHAP explanations

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw CSV files (download from Kaggle)
│   └── processed/            # Cleaned, merged, final datasets
├── models/trained_models/    # Trained models, preprocessor, features
├── src/
│   ├── data/                 # collect_and_merge.py, validate_dataset.py
│   ├── preprocessing/        # pipeline.py
│   ├── features/             # feature_engine
│   ├── modeling/             # train_model.py, evaluate.py
│   └── deployment/           # analyze_app.py, predict_app.py
├── .gitignore
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Setup & Installation

```bash
# Clone repo
git clone https://github.com/Reemsoliiman/customer-churn-prediction-analysis.git
cd customer-churn-prediction-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Download Data

1. Go to: [https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
2. Download:
   - `churn-bigml-80.csv`
   - `churn-bigml-20.csv`
3. Place both files in:  
   `data/raw/`

---

## Run Pipeline (Step-by-Step)

```bash
# 1. Validate data
python src/data/validate_dataset.py

# 2. Merge datasets
python src/data/collect_and_merge_data.py

# 3. Preprocess & encode
python src/preprocessing/pipeline.py

# 4. Feature engineering + selection
python src/features/feature_engineering.py

# 5. Train models
python src/modeling/train_model.py

# 6. Evaluate & save best model
python src/modeling/evaluate.py
```

> All outputs saved automatically.

---

## Launch Apps

### Analysis Dashboard
```bash
streamlit run src/deployment/analyze_app.py
```

### Live Prediction App
```bash
streamlit run src/deployment/predict_app.py
```

---

## Key Features

| Feature | Description |
|-------|-----------|
| **Automated Pipeline** | 6 modular scripts |
| **Feature Selection** | RFE + Random Forest |
| **Model Comparison** | 4 models, best saved |
| **SHAP Explanations** | In prediction app |
| **Interactive Viz** | Plotly + Streamlit |

---

## Model Performance (Example)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|------|----------|---------|--------|--------|
| XGBoost | 0.96 | 0.93 | 0.78 | **0.98** |

> *Best model auto-selected and deployed*

---

## Contributing

1. Fork the repo
2. Create feature branch
3. Commit changes
4. Push & open PR

---

**Built with** Python, Scikit-learn, XGBoost, Streamlit, SHAP, Plotly
```