# Testing Documentation

This directory contains comprehensive tests for the churn prediction project.

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── unit/                            # Unit tests
│   ├── test_helpers.py             # Tests for utility functions
│   └── test_api.py                 # Tests for API schemas and logic
├── integration/                     # Integration tests
│   ├── test_pipeline.py            # End-to-end pipeline tests
│   └── test_api_integration.py     # API integration tests
└── README.md                        # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests by marker
pytest -m unit
pytest -m integration
pytest -m api
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov=api --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Tests in Parallel

```bash
pytest -n auto
```

Uses all available CPU cores.

### Run Specific Test Files or Functions

```bash
# Run specific file
pytest tests/unit/test_helpers.py

# Run specific test class
pytest tests/unit/test_helpers.py::TestEngineerFeatures

# Run specific test function
pytest tests/unit/test_helpers.py::TestEngineerFeatures::test_creates_expected_features
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

## Test Categories

### Unit Tests

**Purpose**: Test individual functions and components in isolation.

**Location**: `tests/unit/`

**Files**:
- `test_helpers.py` - Tests for utility functions (feature engineering, preprocessing, etc.)
- `test_api.py` - Tests for API schemas and prediction logic

**Example**:
```python
def test_engineer_features_creates_total_minutes(sample_data):
    result = engineer_features(sample_data)
    assert "Total_Minutes" in result.columns
```

### Integration Tests

**Purpose**: Test interactions between multiple components and end-to-end workflows.

**Location**: `tests/integration/`

**Files**:
- `test_pipeline.py` - Tests for complete data processing and model training pipelines
- `test_api_integration.py` - Tests for FastAPI endpoints with real requests

**Example**:
```python
def test_full_preprocessing_pipeline(sample_raw_data):
    # Test complete flow: raw data → cleaned → engineered → model ready
    ...
```

## Test Fixtures

Fixtures are defined in `conftest.py` and provide reusable test data and utilities:

- `sample_raw_data` - Mock raw churn dataset
- `sample_processed_data` - Preprocessed data with encoded features
- `sample_engineered_data` - Data with engineered features
- `mock_model_artifacts` - Mock trained model and artifacts
- `sample_api_input` - Valid API input payload
- `temp_dir` - Temporary directory for test artifacts

## Writing New Tests

### Unit Test Template

```python
import pytest
from src.utils.helpers import your_function

class TestYourFunction:
    """Tests for your_function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = your_function(edge_case_input)
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Integration Test Template

```python
import pytest
from pathlib import Path

class TestYourPipeline:
    """Integration tests for your pipeline."""
    
    def test_end_to_end_flow(self, sample_data, temp_dir):
        """Test complete flow from input to output."""
        # Setup
        input_path = temp_dir / "input.csv"
        sample_data.to_csv(input_path, index=False)
        
        # Execute pipeline
        result = run_pipeline(input_path)
        
        # Verify
        assert result is not None
        assert result.shape[0] > 0
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Use Fixtures**: Leverage fixtures for common setup and teardown
3. **Clear Names**: Use descriptive test names that explain what is being tested
4. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
5. **Edge Cases**: Test boundary conditions and error cases
6. **Mock External Dependencies**: Use mocks for external services or file I/O
7. **Fast Tests**: Keep unit tests fast; move slow tests to integration suite

## Continuous Integration

Tests should run automatically in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov=src --cov=api --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the project root is in your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Or run pytest from the project root directory.

### Fixture Not Found

Ensure `conftest.py` is in the correct location and pytest can discover it.

### Tests Pass Locally But Fail in CI

- Check for hardcoded paths
- Ensure all dependencies are in requirements-test.txt
- Verify environment variables are set correctly

## Coverage Goals

Target coverage metrics:
- **Overall**: > 80%
- **Utils/Helpers**: > 90%
- **API**: > 85%
- **Models**: > 75%

## Performance Testing

For performance-critical code, use pytest-benchmark:

```python
def test_feature_engineering_performance(benchmark, sample_data):
    result = benchmark(engineer_features, sample_data)
    assert result is not None
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)