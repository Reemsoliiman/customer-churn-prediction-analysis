import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .schemas import ChurnInput
from src.utils.helpers import engineer_features, align_features_for_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# Load model and features
_model = joblib.load(ARTIFACTS_DIR / "best_model_final.pkl")
_selected_features = joblib.load(ARTIFACTS_DIR / "selected_features.pkl")

# Safe explainer creation
try:
    _explainer = shap.TreeExplainer(_model)
except:
    # Fallback for non-tree models
    test_X, _ = joblib.load(ARTIFACTS_DIR / "test_data.pkl")
    reference = test_X[_selected_features].sample(min(100, len(test_X)), random_state=42)
    _explainer = shap.KernelExplainer(_model.predict_proba, reference)


def preprocess_input(raw: ChurnInput) -> pd.DataFrame:
    """Convert API input to model-ready features"""
    data = raw.model_dump(by_alias=True)
    
    # Convert Yes/No to 1/0
    data["International plan"] = 1 if data["International plan"] == "Yes" else 0
    data["Voice mail plan"] = 1 if data["Voice mail plan"] == "Yes" else 0
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Engineer features (creates new columns like Total_Minutes, etc.)
    df_engineered = engineer_features(df, is_training=False)
    
    # Align to selected features from training
    X_aligned = align_features_for_prediction(df_engineered)
    
    return X_aligned


def predict_and_explain(X: pd.DataFrame) -> Dict[str, Any]:
    """Make prediction and generate SHAP explanations - ROBUST VERSION"""
    
    X_model = X[_selected_features].copy()
    X_values = X_model.values

    # Get prediction probability
    prob = float(_model.predict_proba(X_values)[0, 1])
    pred = bool(prob > 0.5)

    # === SHAP EXPLANATION - SAFE HANDLING FOR ALL MODEL TYPES ===
    try:
        raw_shap = _explainer.shap_values(X_values)

        # Handle different SHAP output formats
        if isinstance(raw_shap, list):
            # XGBoost / LightGBM / CatBoost style: list of arrays [class0, class1]
            shap_class1 = np.array(raw_shap[1]).flatten()
            base_value = float(_explainer.expected_value[1] if hasattr(_explainer.expected_value, '__len__') else _explainer.expected_value)
        elif raw_shap.ndim == 3:
            # Sometimes shap returns [samples, features, classes]
            shap_class1 = raw_shap[0, :, 1]
            base_value = float(_explainer.expected_value[1])
        else:
            # Standard 2D array (sklearn tree models)
            shap_class1 = raw_shap.flatten() if raw_shap.ndim == 2 else raw_shap[:, 1].flatten()
            base_value = float(_explainer.expected_value)

    except Exception as e:
        # Absolute fallback - should never happen
        shap_class1 = np.zeros(len(_selected_features))
        base_value = 0.0

    # Build SHAP dataframe
    shap_df = pd.DataFrame({
        "feature": _selected_features,
        "shap_value": shap_class1
    })
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False)

    top_features = shap_df.head(6)[['feature', 'shap_value']].round(4).to_dict(orient="records")

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "base_value": round(base_value, 4),
        "top_shap_features": top_features
    }