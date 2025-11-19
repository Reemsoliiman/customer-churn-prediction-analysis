"""
Integration tests for the full ML pipeline.
"""
import pytest
import pandas as pd
from pathlib import Path
import joblib
import shutil


class TestDataProcessingPipeline:
    """Integration tests for data processing pipeline."""
    
    def test_full_preprocessing_pipeline(self, sample_raw_data, temp_dir):
        """Test complete preprocessing pipeline."""
        from src.utils.helpers import (
            handle_missing_values,
            encode_categorical_features
        )
        
        # Save raw data
        raw_path = temp_dir / "raw_data.csv"
        sample_raw_data.to_csv(raw_path, index=False)
        
        # Load and process
        df = pd.read_csv(raw_path)
        df = df.drop_duplicates()
        df = handle_missing_values(df)
        df = encode_categorical_features(df)
        
        # Verify processing
        assert len(df) > 0
        assert "Churn" in df.columns
        assert df["Churn"].dtype in [int, 'int64']
        assert "International plan" in df.columns
        assert df["International plan"].dtype in [int, 'int64']
        
        # Save processed
        processed_path = temp_dir / "processed_data.csv"
        df.to_csv(processed_path, index=False)
        
        assert processed_path.exists()
    
    def test_feature_engineering_pipeline(self, sample_processed_data, temp_dir):
        """Test feature engineering pipeline."""
        from src.utils.helpers import engineer_features
        
        # Engineer features
        y = sample_processed_data["Churn"]
        X = engineer_features(
            sample_processed_data.drop("Churn", axis=1),
            is_training=True
        )
        
        # Verify new features
        assert "Total_Minutes" in X.columns
        assert "Total_Calls" in X.columns
        assert "Customer_tenure_months" in X.columns
        assert "High_Service_Calls" in X.columns
        
        # Verify log features
        log_features = [c for c in X.columns if c.startswith("log_")]
        assert len(log_features) > 0
        
        # Save engineered data
        final_df = X.copy()
        final_df["Churn"] = y
        
        final_path = temp_dir / "engineered_data.csv"
        final_df.to_csv(final_path, index=False)
        
        assert final_path.exists()


class TestModelTrainingPipeline:
    """Integration tests for model training pipeline."""
    
    def test_train_and_save_models(self, sample_engineered_data, temp_dir):
        """Test training and saving multiple models."""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare data
        X = sample_engineered_data.drop("Churn", axis=1)
        y = sample_engineered_data["Churn"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        models_dir = temp_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            model_path = models_dir / f"{name}.pkl"
            joblib.dump(model, model_path)
            
            assert model_path.exists()
            
            # Verify model can be loaded and used
            loaded_model = joblib.load(model_path)
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == len(X_test)
    
    def test_feature_selection_integration(self, sample_engineered_data, temp_dir):
        """Test feature selection with RFE."""
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_engineered_data.drop("Churn", axis=1)
        y = sample_engineered_data["Churn"]
        
        # Apply RFE
        n_features = min(10, len(X.columns))
        rfe = RFE(
            RandomForestClassifier(n_estimators=10, random_state=42),
            n_features_to_select=n_features
        )
        rfe.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        
        assert len(selected_features) == n_features
        
        # Save selected features
        features_path = temp_dir / "selected_features.pkl"
        joblib.dump(selected_features, features_path)
        
        assert features_path.exists()
        
        # Verify can be loaded
        loaded_features = joblib.load(features_path)
        assert loaded_features == selected_features


class TestEndToEndPrediction:
    """End-to-end integration tests for prediction — using the REAL production path."""

    def test_raw_input_to_prediction(self, mock_model_artifacts, sample_api_input):
        import joblib
        from src.api.schemas import ChurnInput
        from src.api.predict import preprocess_input

        model = joblib.load(mock_model_artifacts / "best_model_final.pkl")

        # Wrap the dict in the real Pydantic model
        churn_input = ChurnInput(**sample_api_input)

        # Use the EXACT same preprocessing as the real /predict endpoint
        X_aligned = preprocess_input(churn_input)

        prediction = model.predict(X_aligned)
        probability = model.predict_proba(X_aligned)[0, 1]

        assert len(prediction) == 1
        assert 0 <= probability <= 1
        assert prediction[0] in [0, 1]

    def test_batch_prediction(self, mock_model_artifacts, sample_api_input):
        import joblib
        from src.api.schemas import ChurnInput
        from src.api.predict import preprocess_input

        model = joblib.load(mock_model_artifacts / "best_model_final.pkl")

        # Create 5 different inputs by varying fields
        base_input = sample_api_input.copy()
        batch_inputs = []
        for i in range(5):
            inp = base_input.copy()
            inp["Customer service calls"] = i
            inp["Account length"] = 100 + i * 20
            inp["International plan"] = "Yes" if i % 2 == 0 else "No"
            batch_inputs.append(inp)

        predictions = []
        for raw_dict in batch_inputs:
            churn_input = ChurnInput(**raw_dict)
            X = preprocess_input(churn_input)
            pred = model.predict(X)[0]
            predictions.append(pred)

        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)

class TestDataConsistency:
    """Tests for data consistency across pipeline stages."""
    
    def test_feature_consistency_train_vs_predict(
        self,
        sample_raw_data,
        mock_model_artifacts,
        sample_api_input
    ):
        """Test that features are consistent between training and prediction."""
        from src.utils.helpers import (
            encode_categorical_features,
            engineer_features,
            prepare_raw_input_for_prediction
        )
        
        # Training path
        train_df = sample_raw_data.copy()
        train_df = encode_categorical_features(train_df)
        train_y = train_df["Churn"]
        train_X = engineer_features(train_df.drop("Churn", axis=1), is_training=True)
        train_features = set(train_X.columns)
        
        # Prediction path
        pred_df = prepare_raw_input_for_prediction(sample_api_input)
        pred_df["International plan"] = (pred_df["International plan"] == "Yes").astype(int)
        pred_df["Voice mail plan"] = (pred_df["Voice mail plan"] == "Yes").astype(int)
        pred_X = engineer_features(pred_df, is_training=False)
        pred_features = set(pred_X.columns)
        
        # Check overlap
        common_features = train_features.intersection(pred_features)
        assert len(common_features) > 0
        
        # Key features should be present in both
        key_features = [
            "Total_Minutes",
            "Total_Calls",
            "Customer_tenure_months"
        ]
        
        for feat in key_features:
            assert feat in train_features
            assert feat in pred_features
    
    def test_no_data_leakage(self, sample_engineered_data):
        """Test that there's no data leakage in train/test split."""
        from sklearn.model_selection import train_test_split
        
        X = sample_engineered_data.drop("Churn", axis=1)
        y = sample_engineered_data["Churn"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check no overlap in indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0
        
        # Check sizes
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)