"""
Model prediction interface.

Provides a single application-facing interface for loading forecasting
models and generating predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.sales_forecast_system.models.lightgbm_model import LightGBMModel
from src.sales_forecast_system.models.prophet_model import ProphetModel
from src.sales_forecast_system.models.xgboost_model import XGBoostModel


class ForecastPredictor:
    """Load trained models and generate forecasts."""

    SUPPORTED_MODELS = {
        "prophet",
        "xgboost",
        "lightgbm",
    }

    def __init__(
        self,
        model_dir: str = "artifacts/trained_models",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}

    def load_models(self) -> Dict[str, Any]:
        """Load all available forecasting models."""

        self.models = {}

        model_loaders = {
            "prophet": (
                ProphetModel,
                self.model_dir / "prophet_model.pkl",
            ),
            "xgboost": (
                XGBoostModel,
                self.model_dir / "xgboost_model.pkl",
            ),
            "lightgbm": (
                LightGBMModel,
                self.model_dir / "lightgbm_model.pkl",
            ),
        }

        for name, (model_class, path) in model_loaders.items():
            if not path.exists():
                continue

            try:
                model = model_class()
                model.load_model(str(path))
                self.models[name] = model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {name} model from {path}: {exc}"
                ) from exc

        if not self.models:
            raise FileNotFoundError(
                f"No trained models found in {self.model_dir}"
            )

        return self.models

    def available_models(self) -> list[str]:
        """Return loaded model names."""

        return sorted(self.models.keys())

    def predict(
        self,
        model_name: str,
        data: pd.DataFrame,
        periods: int = 12,
    ) -> pd.DataFrame:
        """Generate predictions using a selected model."""

        if periods < 1:
            raise ValueError("periods must be greater than zero")

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )

        if model_name not in self.models:
            self.load_models()

        model = self.models[model_name]

        if model_name == "prophet":
            return self._predict_prophet(model, data, periods)

        raise NotImplementedError(
            f"Application inference for '{model_name}' requires "
            "recursive feature generation and is handled by the "
            "forecasting pipeline."
        )

    @staticmethod
    def _predict_prophet(
        model: Any,
        data: pd.DataFrame,
        periods: int,
    ) -> pd.DataFrame:
        """Generate Prophet predictions."""

        required = {"Date", "Weekly_Sales"}

        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing columns for Prophet prediction: "
                f"{sorted(missing)}"
            )

        history = data[["Date", "Weekly_Sales"]].copy()
        history["Date"] = pd.to_datetime(history["Date"])
        history = history.sort_values("Date")

        # Retrain on the complete historical series before forecasting.
        model.train(history)

        forecast = model.predict(periods)

        if isinstance(forecast, pd.DataFrame):
            result = forecast.copy()
        else:
            result = pd.DataFrame({"prediction": forecast})

        if "yhat" in result.columns:
            result["prediction"] = result["yhat"]

        if "ds" in result.columns:
            result["date"] = pd.to_datetime(result["ds"])

        return result
