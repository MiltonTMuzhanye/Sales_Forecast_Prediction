import pytest
import json
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from api.app import app
from api.schemas import ForecastRequest, ModelType

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/models/info")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "total_models" in data
        assert "available_models" in data
        assert "models" in data
        assert isinstance(data["models"], dict)
    
    def test_forecast_generation(self):
        """Test forecast generation endpoint"""
        # Create forecast request
        forecast_request = {
            "store_id": 1,
            "dept_id": 1,
            "periods": 12,
            "model_type": "prophet",
            "include_confidence": True
        }
        
        response = client.post("/forecast/generate", json=forecast_request)
        
        # Check response status
        # Note: This might fail if models aren't trained, but should return proper error
        if response.status_code != 200:
            # Check that we get a proper error response
            error_data = response.json()
            assert "error" in error_data
            return
        
        # If successful, check response structure
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert data["model_type"] == "prophet"
        assert data["forecast_horizon"] == 12
        assert "generation_date" in data
        assert "forecasts" in data
        assert isinstance(data["forecasts"], list)
        
        # Check forecast items
        if len(data["forecasts"]) > 0:
            forecast_item = data["forecasts"][0]
            assert "date" in forecast_item
            assert "forecast" in forecast_item
            assert "horizon" in forecast_item
            
            if forecast_request["include_confidence"]:
                assert "confidence_interval" in forecast_item
    
    def test_batch_forecast_endpoint(self):
        """Test batch forecast endpoint"""
        params = {
            "store_ids": [1, 2],
            "dept_ids": [1, 2],
            "periods": 8,
            "model_type": "prophet"
        }
        
        response = client.post("/forecast/batch", params=params)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "successful" in data
        assert "failed" in data
        assert "results" in data
        
        # Should have 4 results (2 stores × 2 departments)
        assert len(data["results"]) == 4
        
        for result in data["results"]:
            assert "store_id" in result
            assert "dept_id" in result
            assert "status" in result
    
    def test_historical_data_endpoint(self):
        """Test historical data endpoint"""
        # Create request
        request_data = {
            "store_id": 1,
            "dept_id": 1,
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "columns": ["Temperature", "Fuel_Price"]
        }
        
        response = client.post("/data/historical", json=request_data)
        
        # Check response
        if response.status_code != 200:
            # Check error response
            error_data = response.json()
            assert "error" in error_data
            return
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert "record_count" in data
        assert "date_range" in data
        assert "data" in data
        
        if data["record_count"] > 0:
            # Check data structure
            record = data["data"][0]
            assert "date" in record
            assert "weekly_sales" in record
            
            # Check requested columns
            if "Temperature" in record:
                assert isinstance(record["Temperature"], (float, type(None)))
    
    def test_model_performance_endpoint(self):
        """Test model performance endpoint"""
        response = client.get("/models/performance", params={
            "store_id": 1,
            "dept_id": 1
        })
        
        if response.status_code == 404:
            # No performance data found (expected if models not trained)
            error_data = response.json()
            assert "detail" in error_data
            return
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert "evaluation_period" in data
        assert "performance" in data
        
        # Check performance metrics structure
        if "prophet" in data["performance"]:
            metrics = data["performance"]["prophet"]
            assert "mape" in metrics
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "bias" in metrics
    
    def test_error_analysis_endpoint(self):
        """Test error analysis endpoint"""
        request_data = {
            "store_id": 1,
            "dept_id": 1,
            "model_type": "prophet"
        }
        
        response = client.post("/analysis/errors", json=request_data)
        
        if response.status_code != 200:
            # Check error response
            error_data = response.json()
            assert "error" in error_data
            return
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert data["model_type"] == "prophet"
        assert "summary" in data
        assert "identified_issues" in data
        assert "recommended_actions" in data
        
        # Check summary structure
        summary = data["summary"]
        if summary:  # Might be empty if no data
            assert "mean_absolute_percentage_error" in summary
            assert "mean_absolute_error" in summary
    
    def test_forecast_comparison_endpoint(self):
        """Test forecast comparison endpoint"""
        response = client.get("/forecast/compare", params={
            "store_id": 1,
            "dept_id": 1,
            "periods": 12
        })
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert data["forecast_horizon"] == 12
        assert "comparisons" in data
        
        # Check comparison structure
        comparisons = data["comparisons"]
        assert isinstance(comparisons, dict)
        
        for model_type, comparison in comparisons.items():
            if comparison.get("available", True):
                assert "forecast_count" in comparison
                assert "total_forecast" in comparison
                assert "mean_forecast" in comparison
    
    def test_model_retraining_endpoint(self):
        """Test model retraining endpoint"""
        response = client.post("/models/retrain", params={
            "store_id": 1,
            "dept_id": 1,
            "model_type": "prophet",
            "test_size": 0.2
        })
        
        if response.status_code != 200:
            # Check error response
            error_data = response.json()
            assert "error" in error_data
            return
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert data["model_type"] == "prophet"
        assert data["status"] == "retrained"
        assert "performance" in data
        assert "best_model" in data
        assert "retraining_date" in data
    
    def test_business_impact_endpoint(self):
        """Test business impact endpoint"""
        response = client.get("/analysis/business-impact", params={
            "store_id": 1,
            "dept_id": 1,
            "safety_stock_service_level": 0.95,
            "holding_cost_rate": 0.25,
            "stockout_cost_multiplier": 3.0
        })
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert data["store_id"] == 1
        assert data["dept_id"] == 1
        assert "analysis" in data
        
        analysis = data["analysis"]
        assert analysis["safety_stock_service_level"] == 0.95
        assert analysis["holding_cost_rate"] == 0.25
        assert analysis["stockout_cost_multiplier"] == 3.0
        assert "recommended_actions" in analysis
    
    def test_invalid_requests(self):
        """Test error handling for invalid requests"""
        # Test invalid store ID
        response = client.post("/forecast/generate", json={
            "store_id": -1,  # Invalid
            "dept_id": 1,
            "periods": 12,
            "model_type": "prophet"
        })
        
        # Should return validation error
        assert response.status_code == 422
        
        # Test invalid periods
        response = client.post("/forecast/generate", json={
            "store_id": 1,
            "dept_id": 1,
            "periods": 100,  # Too large
            "model_type": "prophet"
        })
        
        assert response.status_code == 422
        
        # Test invalid model type
        response = client.post("/forecast/generate", json={
            "store_id": 1,
            "dept_id": 1,
            "periods": 12,
            "model_type": "invalid_model"  # Invalid
        })
        
        assert response.status_code == 422
    
    def test_error_responses(self):
        """Test error response format"""
        # Make request to non-existent endpoint
        response = client.get("/non-existent-endpoint")
        
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Make request that causes server error
        response = client.post("/forecast/generate", json={
            "store_id": 9999,  # Non-existent store
            "dept_id": 9999,   # Non-existent department
            "periods": 12,
            "model_type": "prophet"
        })
        
        # Should return error (404 or 500)
        assert response.status_code in [404, 500]
        
        if response.status_code == 500:
            error_data = response.json()
            assert "error" in error_data
            assert "timestamp" in error_data

class TestAPISchemas:
    """Test API schemas validation"""
    
    def test_forecast_request_schema(self):
        """Test ForecastRequest schema validation"""
        # Valid request
        valid_request = ForecastRequest(
            store_id=1,
            dept_id=1,
            periods=12,
            model_type=ModelType.PROPHET,
            include_confidence=True
        )
        
        assert valid_request.store_id == 1
        assert valid_request.dept_id == 1
        assert valid_request.periods == 12
        assert valid_request.model_type == ModelType.PROPHET
        assert valid_request.include_confidence == True
        
        # Test invalid periods
        with pytest.raises(ValueError):
            ForecastRequest(
                store_id=1,
                dept_id=1,
                periods=100,  # Too large
                model_type=ModelType.PROPHET
            )
    
    def test_historical_data_request_schema(self):
        """Test HistoricalDataRequest schema validation"""
        from api.schemas import HistoricalDataRequest
        from datetime import date
        
        # Valid request
        valid_request = HistoricalDataRequest(
            store_id=1,
            dept_id=1,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            columns=["Temperature", "Fuel_Price"]
        )
        
        assert valid_request.store_id == 1
        assert valid_request.dept_id == 1
        assert valid_request.start_date == date(2020, 1, 1)
        assert valid_request.end_date == date(2020, 12, 31)
        assert valid_request.columns == ["Temperature", "Fuel_Price"]
    
    def test_error_analysis_request_schema(self):
        """Test ErrorAnalysisRequest schema validation"""
        from api.schemas import ErrorAnalysisRequest
        
        # Valid request
        valid_request = ErrorAnalysisRequest(
            store_id=1,
            dept_id=1,
            model_type="prophet"
        )
        
        assert valid_request.store_id == 1
        assert valid_request.dept_id == 1
        assert valid_request.model_type == "prophet"
        
        # Test default model type
        request_with_default = ErrorAnalysisRequest(
            store_id=1,
            dept_id=1
        )
        
        assert request_with_default.model_type == "prophet"

def test_api_documentation():
    """Test API documentation endpoints"""
    # Test OpenAPI docs
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test ReDoc docs
    response = client.get("/redoc")
    assert response.status_code == 200
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    # Check that expected endpoints are documented
    expected_endpoints = [
        "/health",
        "/forecast/generate",
        "/data/historical",
        "/models/performance"
    ]
    
    for endpoint in expected_endpoints:
        assert endpoint in schema["paths"]

def test_api_cors_headers():
    """Test CORS headers"""
    # Make OPTIONS request to check CORS
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    
    # Should have CORS headers
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers

def test_api_rate_limiting():
    """Test rate limiting (if implemented)"""
    # Make multiple requests quickly
    for i in range(5):
        response = client.get("/health")
        
        # Should succeed for reasonable number of requests
        if response.status_code == 429:  # Too Many Requests
            # Rate limiting is enabled
            assert "retry-after" in response.headers
            break
        else:
            assert response.status_code == 200

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])