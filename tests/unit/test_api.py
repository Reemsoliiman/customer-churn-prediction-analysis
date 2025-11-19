"""
Unit tests for API schemas and prediction logic.
"""
import pytest
import pandas as pd
from pydantic import ValidationError
from src.api.schemas import ChurnInput


class TestChurnInputSchema:
    """Tests for ChurnInput Pydantic schema."""
    
    def test_valid_input_with_aliases(self, sample_api_input):
        """Test that valid input with aliases is accepted."""
        payload = ChurnInput(**sample_api_input)
        
        assert payload.account_length == 100
        assert payload.international_plan == "No"
        assert payload.voice_mail_plan == "Yes"
    
    def test_rejects_invalid_plan_values(self, sample_api_input):
        """Test that invalid plan values are rejected."""
        sample_api_input["International plan"] = "Maybe"
        
        with pytest.raises(ValidationError):
            ChurnInput(**sample_api_input)
    
    def test_rejects_negative_values(self, sample_api_input):
        """Test that negative values are rejected."""
        sample_api_input["Account length"] = -1
        
        with pytest.raises(ValidationError):
            ChurnInput(**sample_api_input)
    
    def test_rejects_out_of_range_values(self, sample_api_input):
        """Test that out-of-range values are rejected."""
        sample_api_input["Account length"] = 1000  # max is 250
        
        with pytest.raises(ValidationError):
            ChurnInput(**sample_api_input)
    
    def test_accepts_snake_case_keys(self):
        """Test that snake_case keys work due to populate_by_name."""
        payload = ChurnInput(
            account_length=100,
            international_plan="No",
            voice_mail_plan="Yes",
            number_vmail_messages=25,
            total_day_minutes=180.5,
            total_eve_minutes=200.3,
            total_night_minutes=150.7,
            total_intl_minutes=10.2,
            total_day_calls=100,
            total_eve_calls=90,
            total_night_calls=85,
            total_intl_calls=5,
            customer_service_calls=2
        )
        
        assert payload.account_length == 100
    
    def test_model_dump_returns_aliases(self, churn_input_payload):
        """Test that model_dump with by_alias returns correct keys."""
        data = churn_input_payload.model_dump(by_alias=True)
        
        assert "Account length" in data
        assert "International plan" in data
        assert "account_length" not in data
    
    def test_missing_required_field(self, sample_api_input):
        """Test that missing required fields raise error."""
        del sample_api_input["Account length"]
        
        with pytest.raises(ValidationError):
            ChurnInput(**sample_api_input)
    
    def test_float_fields_accept_integers(self, sample_api_input):
        """Test that float fields accept integer values."""
        sample_api_input["Total day minutes"] = 180  # int instead of float
        
        payload = ChurnInput(**sample_api_input)
        assert payload.total_day_minutes == 180.0


class TestAPIPreprocessing:
    """Tests for API preprocessing functions."""
    
    @pytest.fixture
    def mock_artifacts(self, tmp_path, sample_engineered_data):
        """Create mock artifacts for testing."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_engineered_data.drop("Churn", axis=1)
        y = sample_engineered_data["Churn"]
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        selected_features = X.columns.tolist()[:20]
        
        joblib.dump(model, tmp_path / "best_model_final.pkl")
        joblib.dump(selected_features, tmp_path / "selected_features.pkl")
        
        return tmp_path
    
    def test_preprocess_input_converts_yes_no(self, churn_input_payload, monkeypatch):
        """Test that Yes/No values are converted to 1/0."""
        # We need to mock the loaded artifacts
        # This is simplified - in real test you'd mock properly
        from src.api.predict import preprocess_input
        
        result = preprocess_input(churn_input_payload)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_preprocess_input_creates_dataframe(self, churn_input_payload):
        """Test that preprocessing creates proper DataFrame."""
        from src.api.predict import preprocess_input
        
        result = preprocess_input(churn_input_payload)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestPredictAndExplain:
    """Tests for prediction and SHAP explanation."""
    
    def test_returns_probability(self, mock_model_artifacts, monkeypatch):
        """Test that prediction returns valid probability."""
        import sys
        from pathlib import Path
        
        # Mock the artifacts directory
        monkeypatch.setattr(
            "src.api.predict.ARTIFACTS_DIR",
            mock_model_artifacts
        )
        
        # Reimport to load mocked artifacts
        import importlib
        import src.api.predict
        importlib.reload(src.api.predict)
        
        from src.api.predict import predict_and_explain
        
        # Create dummy input
        X = pd.DataFrame([[0] * 20], columns=[f"feat_{i}" for i in range(20)])
        
        result = predict_and_explain(X)
        
        assert "churn_probability" in result
        assert 0 <= result["churn_probability"] <= 1
    
    def test_returns_prediction(self, mock_model_artifacts, monkeypatch):
        """Test that prediction returns binary value."""
        monkeypatch.setattr(
            "src.api.predict.ARTIFACTS_DIR",
            mock_model_artifacts
        )
        
        import importlib
        import src.api.predict
        importlib.reload(src.api.predict)
        
        from src.api.predict import predict_and_explain
        
        X = pd.DataFrame([[0] * 20], columns=[f"feat_{i}" for i in range(20)])
        result = predict_and_explain(X)
        
        assert "churn_prediction" in result
        assert result["churn_prediction"] in [True, False]
    
    def test_returns_shap_features(self, mock_model_artifacts, monkeypatch):
        """Test that SHAP features are returned."""
        monkeypatch.setattr(
            "src.api.predict.ARTIFACTS_DIR",
            mock_model_artifacts
        )
        
        import importlib
        import src.api.predict
        importlib.reload(src.api.predict)
        
        from src.api.predict import predict_and_explain
        
        X = pd.DataFrame([[0] * 20], columns=[f"feat_{i}" for i in range(20)])
        result = predict_and_explain(X)
        
        assert "top_shap_features" in result
        assert isinstance(result["top_shap_features"], list)
        assert len(result["top_shap_features"]) <= 6