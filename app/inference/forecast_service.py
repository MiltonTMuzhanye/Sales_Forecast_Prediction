"""
Application forecasting service.

This module provides the business-facing service used by the API
and frontend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd
import yaml

from app.inference.preprocess import InferencePreprocessor
from src.sales_forecast_system.pipelines.forecasting_pipeline import (
    ForecastingPipeline,
)
from src.sales_forecast_system.utils.config import Config


class ForecastService:
    """High-level forecasting service."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
    ) -> None:
        self.config_path = config_path
        self.config = Config(config_path).config

        # config.py merges data.yaml into config["data"], so raw_path
        # must be read directly from the base config.yaml.
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or {}

        raw_path = base_config.get("data", {}).get(
            "raw_path",
            "data/raw/",
        )

        self.preprocessor = InferencePreprocessor(
            raw_path=raw_path,
            config_path=config_path,
        )

        self.pipeline = ForecastingPipeline(self.config)

    def forecast(
        self,
        store_id: int,
        dept_id: int,
        periods: int = 12,
        model: str = "prophet",
    ) -> Dict[str, Any]:
        """Generate a forecast for a Store/Department."""

        if periods < 1:
            raise ValueError("periods must be greater than zero")

        if periods > 104:
            raise ValueError(
                "periods cannot exceed 104 weekly periods"
            )

        if store_id < 1:
            raise ValueError("store_id must be positive")

        if dept_id < 1:
            raise ValueError("dept_id must be positive")

        allowed_models = {
            "prophet",
            "xgboost",
            "lightgbm",
            "hybrid",
        }

        if model not in allowed_models:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose from {sorted(allowed_models)}"
            )

        result = self.pipeline.forecast(
            store_id=store_id,
            dept_id=dept_id,
            periods=periods,
            model_name=model,
        )

        return self._normalize_result(
            result=result,
            store_id=store_id,
            dept_id=dept_id,
            model=model,
            periods=periods,
        )

    @staticmethod
    def _normalize_result(
        result: Any,
        store_id: int,
        dept_id: int,
        model: str,
        periods: int,
    ) -> Dict[str, Any]:
        """Normalize pipeline output for API/frontend consumption."""

        if isinstance(result, dict):
            dates = result.get("dates", [])
            predictions = result.get(
                "predictions",
                result.get("forecast", []),
            )

            return {
                "store_id": store_id,
                "department_id": dept_id,
                "model": model,
                "periods": periods,
                "generated_at": datetime.utcnow().isoformat(),
                "dates": [str(x) for x in dates],
                "predictions": [float(x) for x in predictions],
            }

        if isinstance(result, pd.DataFrame):
            date_column = next(
                (
                    column
                    for column in ["date", "ds", "Date"]
                    if column in result.columns
                ),
                None,
            )

            prediction_column = next(
                (
                    column
                    for column in ["prediction", "yhat", "forecast"]
                    if column in result.columns
                ),
                None,
            )

            if date_column is None or prediction_column is None:
                raise ValueError(
                    "Forecast result does not contain recognizable "
                    "date and prediction columns."
                )

            return {
                "store_id": store_id,
                "department_id": dept_id,
                "model": model,
                "periods": len(result),
                "generated_at": datetime.utcnow().isoformat(),
                "dates": [
                    pd.to_datetime(x).isoformat()
                    for x in result[date_column]
                ],
                "predictions": [
                    float(x)
                    for x in result[prediction_column]
                ],
            }

        raise TypeError(
            f"Unsupported forecast result type: {type(result).__name__}"
        )
