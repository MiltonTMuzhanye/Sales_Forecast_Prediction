import pandas as pd
import pytest

from app.inference.forecast_service import ForecastService
from sales_forecast_system.evaluation.metrics import ModelEvaluator


def test_forecast_service_prophet():
    service = ForecastService()

    result = service.forecast(
        store_id=1,
        dept_id=1,
        periods=4,
        model="prophet",
    )

    assert result["store_id"] == 1
    assert result["department_id"] == 1
    assert result["model"] == "prophet"
    assert result["periods"] == 4
    assert len(result["predictions"]) == 4
    assert len(result["dates"]) == 4

    assert all(
        isinstance(value, (int, float))
        for value in result["predictions"]
    )


def test_forecast_service_validation():
    service = ForecastService()

    with pytest.raises(ValueError):
        service.forecast(
            store_id=0,
            dept_id=1,
            periods=4,
            model="prophet",
        )

    with pytest.raises(ValueError):
        service.forecast(
            store_id=1,
            dept_id=1,
            periods=0,
            model="prophet",
        )

    with pytest.raises(ValueError):
        service.forecast(
            store_id=1,
            dept_id=1,
            periods=105,
            model="prophet",
        )


def test_forecast_service_rejects_unavailable_models():
    service = ForecastService()

    with pytest.raises(ValueError):
        service.forecast(
            store_id=1,
            dept_id=1,
            periods=4,
            model="xgboost",
        )


def test_forecast_dates_are_ordered():
    service = ForecastService()

    result = service.forecast(
        store_id=1,
        dept_id=1,
        periods=8,
        model="prophet",
    )

    dates = pd.to_datetime(result["dates"])

    assert len(dates) == 8
    assert dates.is_monotonic_increasing


def test_metrics_calculation():
    actual = pd.Series([100, 110, 120, 130])
    predicted = pd.Series([105, 108, 118, 125])

    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(actual, predicted)

    assert isinstance(metrics, dict)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert "smape" in metrics
    assert "r2" in metrics

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["mape"] >= 0
