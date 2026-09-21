import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluates model performance."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate regression metrics."""

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Prediction length mismatch: "
                f"y_true has {len(y_true)} values, "
                f"y_pred has {len(y_pred)} values."
            )

        if len(y_true) == 0:
            raise ValueError("Cannot calculate metrics on empty arrays.")

        metrics = {
            "rmse": float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            ),
            "mae": float(
                mean_absolute_error(y_true, y_pred)
            ),
            "mape": float(
                mean_absolute_percentage_error(y_true, y_pred) * 100
            ),
            "smape": float(
                np.mean(
                    2
                    * np.abs(y_pred - y_true)
                    / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
                )
                * 100
            ),
        }

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        metrics["r2"] = float(
            1 - (ss_res / (ss_tot + 1e-8))
        )

        return metrics

    def compare_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """Compare multiple models using the same ground-truth values."""

        results = []

        for model_name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics["model"] = model_name
            results.append(metrics)

        if not results:
            return pd.DataFrame(
                columns=["rmse", "mae", "mape", "smape", "r2"]
            ).rename_axis("model")

        df = pd.DataFrame(results)

        return df.set_index("model")
