import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .schemas import ChurnInput
from src.utils.helpers import engineer_features, align_features_for_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# Load once at import
_model = joblib.load(ARTIFACTS_DIR / "best_model_final.pkl")
_selected_features = joblib.load(ARTIFACTS_DIR / "selected_features.pkl")
_explainer = shap.TreeExplainer(_model)

def preprocess_input(raw: ChurnInput) -> pd.DataFrame:
    data = raw.model_dump(by_alias=True)

    # Convert Yes/No to 1/0
    data["International plan"] = 1 if data["International plan"] == "Yes" else 0
    data["Voice mail plan"] = 1 if data["Voice mail plan"] == "Yes" else 0

    df = pd.DataFrame([data])
    df_engineered = engineer_features(df, is_training=False)
    X_aligned = align_features_for_prediction(df_engineered)
    return X_aligned

def predict_and_explain(X: pd.DataFrame) -> Dict[str, Any]:
    X_np = X.values
    prob = _model.predict_proba(X_np)[0, 1]
    pred = int(prob > 0.5)

    # SHAP
    shap_vals = _explainer.shap_values(X_np)
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1].flatten()
        base_value = _explainer.expected_value[1]
    else:
        shap_class1 = shap_vals.flatten()
        base_value = _explainer.expected_value

    shap_df = pd.DataFrame({
        "feature": _selected_features,
        "shap_value": shap_class1
    }).assign(abs_shap=lambda d: d["shap_value"].abs())\
     .sort_values("abs_shap", ascending=False)

    top_features = shap_df.head(6).to_dict(orient="records")

    return {
        "churn_probability": float(prob),
        "churn_prediction": bool(pred),
        "base_value": float(base_value),
        "top_shap_features": top_features
    }