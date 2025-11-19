"""
Unit tests for utility helper functions.
"""
import pytest
import pandas as pd
import numpy as np
from src.utils.helpers import (
    engineer_features,
    prepare_raw_input_for_prediction,
    align_features_for_prediction,
    clip_outliers_iqr,
    handle_missing_values,
    encode_categorical_features,
    validate_input_data
)


class TestEngineerFeatures:
    """Tests for feature engineering function."""
    
    def test_creates_expected_features(self, sample_raw_data):
        """Test that all expected features are created."""
        df = sample_raw_data.drop("Churn", axis=1).copy()
        result = engineer_features(df, is_training=True)
        
        expected_features = [
            "Customer_tenure_months",
            "Total_Minutes",
            "Total_Calls",
            "Avg_Daily_Usage",
            "Call_Frequency",
            "Intl_Usage_Rate",
            "High_Service_Calls",
            "Has_Vmail"
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_removes_charge_columns(self, sample_raw_data):
        """Test that charge columns are removed."""
        df = sample_raw_data.drop("Churn", axis=1).copy()
        result = engineer_features(df, is_training=True)
        
        charge_cols = [c for c in result.columns if "charge" in c.lower()]
        assert len(charge_cols) == 0, "Charge columns should be removed"
    
    def test_log_transforms_applied(self, sample_raw_data):
        """Test that log transforms are created."""
        df = sample_raw_data.drop("Churn", axis=1).copy()
        result = engineer_features(df, is_training=True)
        
        log_features = [c for c in result.columns if c.startswith("log_")]
        assert len(log_features) > 0, "Log features should be created"
    
    def test_handles_missing_columns_gracefully(self):
        """Test that function handles missing input columns."""
        df = pd.DataFrame({
            "Account length": [100, 150],
            "Total day minutes": [180, 200]
        })
        
        # Should not raise error
        result = engineer_features(df, is_training=True)
        assert len(result) == 2
    
    def test_customer_tenure_calculation(self):
        """Test customer tenure calculation."""
        df = pd.DataFrame({
            "Account length": [30, 60, 90]
        })
        
        result = engineer_features(df, is_training=True)
        assert "Customer_tenure_months" in result.columns
        assert result["Customer_tenure_months"].iloc[0] == pytest.approx(1.0)
        assert result["Customer_tenure_months"].iloc[1] == pytest.approx(2.0)


class TestHandleMissingValues:
    """Tests for missing value imputation."""
    
    def test_imputes_numerical_with_median(self):
        """Test that numerical columns are imputed with median."""
        df = pd.DataFrame({
            "col1": [1, 2, np.nan, 4, 5],
            "col2": [10, np.nan, 30, 40, 50]
        })
        
        result = handle_missing_values(df)
        assert result["col1"].isna().sum() == 0
        assert result["col2"].isna().sum() == 0
        assert result["col1"].iloc[2] == 3.0  # median of [1,2,4,5]
    
    def test_imputes_categorical_with_mode(self):
        """Test that categorical columns are imputed with mode."""
        df = pd.DataFrame({
            "cat_col": ["A", "B", np.nan, "A", "A"]
        })
        
        result = handle_missing_values(df)
        assert result["cat_col"].isna().sum() == 0
        assert result["cat_col"].iloc[2] == "A"  # mode
    
    def test_handles_no_missing_values(self):
        """Test that function works with complete data."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"]
        })
        
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, df)


class TestEncodeCategoricalFeatures:
    """Tests for categorical encoding."""
    
    def test_encodes_binary_yes_no(self):
        """Test Yes/No encoding to 1/0."""
        df = pd.DataFrame({
            "International plan": ["Yes", "No", "Yes"],
            "Voice mail plan": ["No", "Yes", "No"]
        })
        
        result = encode_categorical_features(df)
        assert result["International plan"].dtype in [np.int64, np.int32]
        assert result["Voice mail plan"].dtype in [np.int64, np.int32]
        assert result["International plan"].tolist() == [1, 0, 1]
        assert result["Voice mail plan"].tolist() == [0, 1, 0]
    
    def test_encodes_churn_column(self):
        df = pd.DataFrame({
            "International plan": ["Yes", "No"],
            "Voice mail plan": ["Yes", "No"],   # ← this was missing!
            "Churn": [True, False],
            "State": ["NY", "CA"],
        })
        result = encode_categorical_features(df)
    
        assert result["Churn"].dtype == int
        assert result["Churn"].tolist() == [1, 0]
    
    def test_one_hot_encodes_state(self):
        """Test one-hot encoding for State."""
        df = pd.DataFrame({
            "State": ["CA", "NY", "TX", "CA"]
        })
        
        result = encode_categorical_features(df)
        assert "State" not in result.columns
        state_cols = [c for c in result.columns if c.startswith("State_")]
        assert len(state_cols) > 0
    
    def test_drops_area_code(self):
        """Test that Area code is dropped."""
        df = pd.DataFrame({
            "Area code": [408, 415, 510]
        })
        
        result = encode_categorical_features(df)
        assert "Area code" not in result.columns


class TestClipOutliersIQR:
    """Tests for outlier clipping."""
    
    def test_clips_outliers(self):
        """Test that extreme values are clipped."""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        result = clip_outliers_iqr(data)
        
        assert result.max() < 100
        assert result.min() >= 1
    
    def test_preserves_normal_values(self):
        """Test that normal values are unchanged."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = clip_outliers_iqr(data)
        
        pd.testing.assert_series_equal(result, data)


class TestAlignFeaturesForPrediction:
    """Tests for feature alignment."""
    
    def test_reorders_columns(self):
        """Test that columns are reordered correctly."""
        df = pd.DataFrame({
            "c": [1, 2],
            "a": [3, 4],
            "b": [5, 6]
        })
        
        selected = ["a", "b", "c"]
        result = align_features_for_prediction(df)
        
        assert result.columns.tolist() == selected
    
    def test_fills_missing_columns(self):
        """Test that missing columns are filled with 0."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4]
        })
        
        selected = ["a", "b", "c"]
        result = align_features_for_prediction(df)
        
        assert "c" in result.columns
        assert result["c"].tolist() == [0, 0]
    
    def test_removes_extra_columns(self):
        """Test that extra columns are removed."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6]
        })
        
        selected = ["a", "b"]
        result = align_features_for_prediction(df)
        
        assert "c" not in result.columns


class TestValidateInputData:
    """Tests for input validation."""
    
    def test_passes_with_all_required_columns(self):
        """Test validation passes with complete data."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4]
        })
        
        required = ["col1", "col2"]
        assert validate_input_data(df, required) is True
    
    def test_raises_on_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            "col1": [1, 2]
        })
        
        required = ["col1", "col2"]
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df, required)


class TestPrepareRawInputForPrediction:
    """Tests for raw input preparation."""
    
    def test_converts_dict_to_dataframe(self, sample_api_input):
        """Test conversion of dict to DataFrame."""
        result = prepare_raw_input_for_prediction(sample_api_input)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "Account length" in result.columns
    
    def test_preserves_all_fields(self, sample_api_input):
        """Test that all input fields are preserved."""
        result = prepare_raw_input_for_prediction(sample_api_input)
        
        for key in sample_api_input.keys():
            assert key in result.columns