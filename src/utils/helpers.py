"""
Shared utility functions for feature engineering and data processing.
Used consistently across training pipeline and deployment.
"""
import pandas as pd
import numpy as np
from typing import Union


def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Apply consistent feature engineering to data.
    Used in both training and prediction/deployment.
    
    Args:
        df: DataFrame with raw features (after basic cleaning)
        is_training: If True, expects 'Churn' column; if False, creates features for prediction
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Drop charge columns (redundant with minutes)
    charge_cols = [c for c in df.columns if "charge" in c.lower()]
    df = df.drop(columns=charge_cols, errors="ignore")
    
    # 1. Customer tenure
    if "Account length" in df.columns:
        df["Customer_tenure_months"] = df["Account length"] / 30.0
    
    # 2. Total aggregates
    minute_cols = ["Total day minutes", "Total eve minutes", "Total night minutes", "Total intl minutes"]
    call_cols = ["Total day calls", "Total eve calls", "Total night calls", "Total intl calls"]
    
    # Check if columns exist before summing
    existing_minute_cols = [c for c in minute_cols if c in df.columns]
    existing_call_cols = [c for c in call_cols if c in df.columns]
    
    if existing_minute_cols:
        df["Total_Minutes"] = df[existing_minute_cols].sum(axis=1)
    if existing_call_cols:
        df["Total_Calls"] = df[existing_call_cols].sum(axis=1)
    
    # 3. Usage patterns
    if "Total_Minutes" in df.columns and "Customer_tenure_months" in df.columns:
        df["Avg_Daily_Usage"] = df["Total_Minutes"] / (df["Customer_tenure_months"] + 1)
    
    if "Total_Calls" in df.columns and "Customer_tenure_months" in df.columns:
        df["Call_Frequency"] = df["Total_Calls"] / (df["Customer_tenure_months"] + 1)
    
    if "Total intl minutes" in df.columns and "Total_Minutes" in df.columns:
        df["Intl_Usage_Rate"] = df["Total intl minutes"] / (df["Total_Minutes"] + 1)
    
    # 4. Behavioral flags
    if "Customer service calls" in df.columns:
        df["High_Service_Calls"] = (df["Customer service calls"] > 3).astype(int)
    
    if "Number vmail messages" in df.columns:
        df["Has_Vmail"] = (df["Number vmail messages"] > 0).astype(int)
    
    # 5. Log transforms (with safety)
    log_features = ["Total_Minutes", "Total_Calls", "Avg_Daily_Usage", "Call_Frequency"]
    for col in log_features:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)
            df[f"log_{col}"] = np.log1p(df[col])
    
    return df


def prepare_raw_input_for_prediction(raw_input: dict) -> pd.DataFrame:
    """
    Convert raw user input (from Streamlit form) into a DataFrame
    ready for feature engineering and prediction.
    
    Args:
        raw_input: Dictionary with keys matching form inputs
        
    Returns:
        DataFrame with one row, ready for engineer_features()
    """
    # Create DataFrame
    df = pd.DataFrame([raw_input])
    
    # Handle categorical encoding (if needed)
    # International plan and Voice mail plan should already be 0/1 from form
    
    return df


def load_selected_features(features_path: str) -> list:
    """
    Load the list of selected features from pickle file.
    
    Args:
        features_path: Path to selected_features.pkl
        
    Returns:
        List of feature names
    """
    import joblib
    return joblib.load(features_path)


def align_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align features for prediction using the exact selected_features.pkl from training.
    Works in both production and tests.
    """
    import joblib
    from pathlib import Path

    # Try multiple possible locations
    possible_paths = [
        Path("models/artifacts/selected_features.pkl"),
        Path(__file__).resolve().parent.parent.parent / "models/artifacts/selected_features.pkl",
    ]

    selected_features = None
    for path in possible_paths:
        if path.exists():
            selected_features = joblib.load(path)
            break

    if selected_features is None:
        # Last resort: use the mock in tests
        try:
            import pytest
            if "mock_model_artifacts" in str(path):  # crude but works
                raise FileNotFoundError
        except:
            pass
        raise FileNotFoundError("selected_features.pkl not found")

    return df.reindex(columns=selected_features, fill_value=0)


def clip_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Clip outliers using IQR method.
    
    Args:
        series: Pandas Series to clip
        factor: IQR multiplier (default 1.5)
        
    Returns:
        Clipped Series
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return series.clip(lower_bound, upper_bound)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values: median for numerical, mode for categorical.
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features consistently.
    - Binary Yes/No -> 1/0
    - State -> One-hot encoding (drop first)
    - Churn -> 1/0
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Binary columns
    for col in ["International plan", "Voice mail plan"]:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (df[col] == "Yes").astype(int)
    
    # Churn column
    if "Churn" in df.columns:
        if df[col].dtype == 'object':
            df["Churn"] = (df["Churn"] == True).astype(int)
        elif df["Churn"].dtype == bool:
            df["Churn"] = df["Churn"].astype(int)
    
    # State one-hot encoding
    if "State" in df.columns:
        state_dummies = pd.get_dummies(df["State"], prefix="State", drop_first=True)
        df = pd.concat([df.drop("State", axis=1), state_dummies], axis=1)
    
    # Drop Area code (not predictive)
    df = df.drop("Area code", axis=1, errors="ignore")
    
    return df


def validate_input_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that input DataFrame has all required columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
