"""
Pytest configuration and shared fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import joblib
from sklearn.ensemble import RandomForestClassifier

# tests/conftest.py
import sys
from pathlib import Path

# Add the project root to sys.path
# This makes both 'src' and 'api' importable as packages
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Optional: helpful for debugging
# print(f"Added to PYTHONPATH: {project_root}")

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def sample_raw_data():
    """Create sample raw churn data matching the expected format."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "State": np.random.choice(["NY", "CA", "TX", "FL"], n_samples),
        "Account length": np.random.randint(1, 250, n_samples),
        "Area code": np.random.choice([408, 415, 510], n_samples),
        "International plan": np.random.choice(["Yes", "No"], n_samples),
        "Voice mail plan": np.random.choice(["Yes", "No"], n_samples),
        "Number vmail messages": np.random.randint(0, 50, n_samples),
        "Total day minutes": np.random.uniform(0, 350, n_samples),
        "Total day calls": np.random.randint(0, 200, n_samples),
        "Total day charge": np.random.uniform(0, 60, n_samples),
        "Total eve minutes": np.random.uniform(0, 300, n_samples),
        "Total eve calls": np.random.randint(0, 200, n_samples),
        "Total eve charge": np.random.uniform(0, 50, n_samples),
        "Total night minutes": np.random.uniform(0, 300, n_samples),
        "Total night calls": np.random.randint(0, 200, n_samples),
        "Total night charge": np.random.uniform(0, 25, n_samples),
        "Total intl minutes": np.random.uniform(0, 20, n_samples),
        "Total intl calls": np.random.randint(0, 20, n_samples),
        "Total intl charge": np.random.uniform(0, 5, n_samples),
        "Customer service calls": np.random.randint(0, 9, n_samples),
        "Churn": np.random.choice([True, False], n_samples, p=[0.15, 0.85])
    }
    
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_processed_data(sample_raw_data):
    """Create sample processed data (after encoding)."""
    from src.utils.helpers import encode_categorical_features, handle_missing_values
    
    df = sample_raw_data.copy()
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    
    return df

@pytest.fixture(scope="session")
def sample_engineered_data(sample_processed_data):
    """Create sample data with engineered features."""
    from src.utils.helpers import engineer_features
    
    df = sample_processed_data.copy()
    y = df["Churn"]
    X = engineer_features(df.drop("Churn", axis=1), is_training=True)
    X["Churn"] = y
    
    return X

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)

@pytest.fixture(scope="session")
def mock_model_artifacts(tmp_path_factory, sample_engineered_data):
    """Create mock trained model and artifacts."""
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    
    # Create mock model
    X = sample_engineered_data.drop("Churn", axis=1)
    y = sample_engineered_data["Churn"]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, artifacts_dir / "best_model_final.pkl")
    
    # Save selected features
    selected_features = X.columns.tolist()[:20]  # Top 20 features
    joblib.dump(selected_features, artifacts_dir / "selected_features.pkl")
    
    return artifacts_dir

@pytest.fixture
def sample_api_input():
    """Sample input for API testing."""
    return {
        "Account length": 100,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 25,
        "Total day minutes": 180.5,
        "Total eve minutes": 200.3,
        "Total night minutes": 150.7,
        "Total intl minutes": 10.2,
        "Total day calls": 100,
        "Total eve calls": 90,
        "Total night calls": 85,
        "Total intl calls": 5,
        "Customer service calls": 2
    }

@pytest.fixture
def churn_input_payload():
    """Valid ChurnInput payload for API testing."""
    from src.api.schemas import ChurnInput
    
    return ChurnInput(
        **{
            "Account length": 100,
            "International plan": "No",
            "Voice mail plan": "Yes",
            "Number vmail messages": 25,
            "Total day minutes": 180.5,
            "Total eve minutes": 200.3,
            "Total night minutes": 150.7,
            "Total intl minutes": 10.2,
            "Total day calls": 100,
            "Total eve calls": 90,
            "Total night calls": 85,
            "Total intl calls": 5,
            "Customer service calls": 2
        }
    )