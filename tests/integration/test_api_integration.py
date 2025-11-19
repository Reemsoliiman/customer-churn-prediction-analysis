"""
Integration tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns correct message."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_predict_endpoint_valid_input(self, api_client, sample_api_input):
        """Test prediction endpoint with valid input."""
        response = api_client.post("/predict", json=sample_api_input)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "top_shap_features" in data
        
        # Validate probability range
        assert 0 <= data["churn_probability"] <= 1
        
        # Validate prediction is boolean
        assert isinstance(data["churn_prediction"], bool)
        
        # Validate SHAP features
        assert isinstance(data["top_shap_features"], list)
        assert len(data["top_shap_features"]) <= 6
    
    def test_predict_endpoint_invalid_input(self, api_client):
        """Test prediction endpoint rejects invalid input."""
        invalid_input = {
            "Account length": -1,  # Invalid: negative
            "International plan": "No",
            "Voice mail plan": "Yes"
        }
        
        response = api_client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_field(self, api_client, sample_api_input):
        """Test prediction endpoint rejects incomplete input."""
        # Remove required field
        incomplete_input = sample_api_input.copy()
        del incomplete_input["Account length"]
        
        response = api_client.post("/predict", json=incomplete_input)
        assert response.status_code == 422
    
    def test_predict_endpoint_invalid_plan_value(self, api_client, sample_api_input):
        """Test prediction endpoint rejects invalid plan values."""
        invalid_input = sample_api_input.copy()
        invalid_input["International plan"] = "Maybe"  # Invalid
        
        response = api_client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_endpoint_multiple_requests(self, api_client, sample_api_input):
        """Test multiple sequential predictions."""
        results = []
        
        for i in range(3):
            input_data = sample_api_input.copy()
            input_data["Customer service calls"] = i
            
            response = api_client.post("/predict", json=input_data)
            assert response.status_code == 200
            
            results.append(response.json())
        
        # All should return valid predictions
        for result in results:
            assert "churn_probability" in result
            assert 0 <= result["churn_probability"] <= 1
    
    def test_predict_response_format(self, api_client, sample_api_input):
        """Test that prediction response has correct format."""
        response = api_client.post("/predict", json=sample_api_input)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "churn_probability",
            "churn_prediction",
            "base_value",
            "top_shap_features"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check SHAP feature structure
        for feature in data["top_shap_features"]:
            assert "feature" in feature
            assert "shap_value" in feature
            assert "abs_shap" in feature
    
    def test_predict_high_risk_customer(self, api_client):
        """Test prediction for high-risk customer profile."""
        high_risk_input = {
            "Account length": 50,
            "International plan": "Yes",
            "Voice mail plan": "No",
            "Number vmail messages": 0,
            "Total day minutes": 300.0,
            "Total eve minutes": 250.0,
            "Total night minutes": 200.0,
            "Total intl minutes": 15.0,
            "Total day calls": 150,
            "Total eve calls": 120,
            "Total night calls": 100,
            "Total intl calls": 8,
            "Customer service calls": 5  # High service calls
        }
        
        response = api_client.post("/predict", json=high_risk_input)
        assert response.status_code == 200
        
        data = response.json()
        # Should have some probability (not necessarily high)
        assert 0 <= data["churn_probability"] <= 1
    
    def test_predict_low_risk_customer(self, api_client):
        """Test prediction for low-risk customer profile."""
        low_risk_input = {
            "Account length": 200,
            "International plan": "No",
            "Voice mail plan": "Yes",
            "Number vmail messages": 30,
            "Total day minutes": 150.0,
            "Total eve minutes": 120.0,
            "Total night minutes": 100.0,
            "Total intl minutes": 5.0,
            "Total day calls": 80,
            "Total eve calls": 70,
            "Total night calls": 65,
            "Total intl calls": 3,
            "Customer service calls": 1  # Low service calls
        }
        
        response = api_client.post("/predict", json=low_risk_input)
        assert response.status_code == 200
        
        data = response.json()
        assert 0 <= data["churn_probability"] <= 1


class TestAPIErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_endpoint(self, api_client):
        """Test that invalid endpoints return 404."""
        response = api_client.get("/invalid_endpoint")
        assert response.status_code == 404
    
    def test_wrong_http_method(self, api_client):
        """Test that GET on /predict returns error."""
        response = api_client.get("/predict")
        assert response.status_code == 405  # Method not allowed
    
    def test_malformed_json(self, api_client):
        """Test that malformed JSON is handled."""
        response = api_client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestAPIConcurrency:
    """Tests for API concurrency handling."""
    
    def test_concurrent_predictions(self, api_client, sample_api_input):
        """Test that API handles concurrent requests correctly."""
        import concurrent.futures
        
        def make_request(i):
            input_data = sample_api_input.copy()
            input_data["Customer service calls"] = i % 5
            response = api_client.post("/predict", json=input_data)
            return response.status_code, response.json()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        for status_code, data in results:
            assert status_code == 200
            assert "churn_probability" in data