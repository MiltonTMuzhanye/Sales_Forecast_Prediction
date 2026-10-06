"""
Inference preprocessing utilities.

Loads the project's raw sales data and applies the same preprocessing
logic used by the training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.sales_forecast_system.data.ingestion import DataIngestion
from src.sales_forecast_system.data.preprocessing import DataPreprocessor
from src.sales_forecast_system.utils.config import Config


class InferencePreprocessor:
    """Prepare Store/Department data for inference."""

    def __init__(
        self,
        raw_path: str = "data/raw/",
        config_path: str = "configs/config.yaml",
    ) -> None:
        self.raw_path = Path(raw_path)
        self.config_path = config_path

        self.config = Config(config_path)

        # The shared Config class currently overwrites config["data"]
        # with data.yaml. DataIngestion therefore needs an explicit
        # raw-path override for inference.
        self.ingestion = DataIngestion(self.config)

        self.preprocessor = DataPreprocessor(self.config)

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load all raw data and apply project preprocessing."""

        train = self.ingestion.load_train_data()
        stores = self.ingestion.load_stores_data()
        features = self.ingestion.load_features_data()

        processed = self.preprocessor.preprocess(
            train,
            stores,
            features,
        )

        return processed

    def get_series(
        self,
        store_id: int,
        dept_id: int,
    ) -> pd.DataFrame:
        """Return preprocessed data for one Store/Department series."""

        df = self.load_and_preprocess()

        series = df[
            (df["Store"] == store_id)
            & (df["Dept"] == dept_id)
        ].copy()

        if series.empty:
            raise ValueError(
                f"No data found for Store={store_id}, Dept={dept_id}"
            )

        return series.sort_values("Date").reset_index(drop=True)

    def validate_series(
        self,
        store_id: int,
        dept_id: int,
    ) -> Tuple[bool, str]:
        """Validate that a Store/Department series exists."""

        try:
            series = self.get_series(store_id, dept_id)

            if series.empty:
                return False, "Series is empty"

            if "Date" not in series.columns:
                return False, "Date column is missing"

            if "Weekly_Sales" not in series.columns:
                return False, "Weekly_Sales column is missing"

            if series["Weekly_Sales"].isna().any():
                return False, "Weekly_Sales contains missing values"

            return True, "Series is valid"

        except Exception as exc:
            return False, str(exc)
