from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


class TestAPI:

    def test_root_health(self):
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["models_loaded"] == ["prophet"]

    def test_models_endpoint(self):
        response = client.get("/models")

        assert response.status_code == 200

        data = response.json()

        assert data["available_models"] == ["prophet"]
        assert "training_models" in data
        assert set(data["training_models"]) == {
            "prophet",
            "xgboost",
            "lightgbm",
            "hybrid",
        }

    def test_openapi_routes(self):
        response = client.get("/openapi.json")

        assert response.status_code == 200

        paths = response.json()["paths"]

        assert "/" in paths
        assert "/forecast" in paths
        assert "/forecast/store" in paths
        assert "/forecast/batch" in paths
        assert "/models" in paths

    def test_forecast_endpoint(self):
        response = client.post(
            "/forecast",
            json={
                "store_id": 1,
                "department_id": 1,
                "periods": 12,
                "model": "prophet",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["store_id"] == 1
        assert data["department_id"] == 1
        assert data["model"] == "prophet"
        assert data["periods"] == 12
        assert len(data["predictions"]) == 12
        assert len(data["dates"]) == 12

    def test_forecast_invalid_store(self):
        response = client.post(
            "/forecast",
            json={
                "store_id": 9999,
                "department_id": 9999,
                "periods": 12,
                "model": "prophet",
            },
        )

        assert response.status_code in {400, 500}

    def test_non_prophet_model_rejected(self):
        response = client.post(
            "/forecast",
            json={
                "store_id": 1,
                "department_id": 1,
                "periods": 12,
                "model": "xgboost",
            },
        )

        assert response.status_code == 400

    def test_forecast_period_validation(self):
        response = client.post(
            "/forecast",
            json={
                "store_id": 1,
                "department_id": 1,
                "periods": 0,
                "model": "prophet",
            },
        )

        assert response.status_code == 422

    def test_batch_forecast_endpoint(self):
        response = client.post(
            "/forecast/batch",
            json={
                "stores_depts": {
                    "1": [1],
                },
                "periods": 4,
                "model": "prophet",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "1_1" in data

        result = data["1_1"]

        assert result["store_id"] == 1
        assert result["department_id"] == 1
        assert result["model"] == "prophet"
        assert result["periods"] == 4
        assert len(result["predictions"]) == 4

    def test_cors(self):
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert "access-control-allow-origin" in response.headers
