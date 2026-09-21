import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
import joblib

logger = setup_logger(__name__)


class LightGBMModel:
    """LightGBM model wrapper."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.feature_cols = None
        self.target_col = self.config.get(
            'data.target_column',
            'Weekly_Sales'
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> tuple:
        """Prepare numeric features and target for LightGBM."""

        logger.info("Preparing data for LightGBM...")

        target_col = target_col or self.target_col

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataframe."
            )

        feature_cols = [
            col
            for col in df.columns
            if col != target_col
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not feature_cols:
            raise ValueError(
                "No numeric feature columns available for LightGBM."
            )

        self.target_col = target_col
        self.feature_cols = feature_cols

        X = df[feature_cols].copy().fillna(0)
        y = df[target_col].values

        return X, y

    def build_model(self, **kwargs) -> lgb.LGBMRegressor:
        """Build LightGBM model."""

        logger.info("Building LightGBM model...")

        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1,
        }

        params.update(kwargs)

        return lgb.LGBMRegressor(**params)

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Train LightGBM using a chronological train/test split."""

        logger.info("Training LightGBM model...")

        try:
            if not 0 < test_size < 1:
                raise ValueError(
                    "test_size must be between 0 and 1."
                )

            X, y = self.prepare_data(
                df,
                target_col=target_col
            )

            split_index = int(len(X) * (1 - test_size))

            if split_index <= 0 or split_index >= len(X):
                raise ValueError(
                    f"Invalid chronological split for {len(X)} rows "
                    f"and test_size={test_size}."
                )

            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y[:split_index]
            y_test = y[split_index:]

            logger.info(
                f"Chronological split: "
                f"{len(X_train)} training rows, "
                f"{len(X_test)} validation rows"
            )

            self.model = self.build_model(**kwargs)

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
            )

            logger.info("LightGBM model trained successfully")

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train LightGBM model: {e}"
            )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""

        if self.model is None:
            raise ValueError("Model not trained")

        if self.feature_cols is None:
            raise ValueError("Feature columns are not available")

        missing_features = [
            col
            for col in self.feature_cols
            if col not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing features for prediction: {missing_features}"
            )

        X = df[self.feature_cols].copy().fillna(0)

        return self.model.predict(X)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Evaluate model."""

        return {
            'rmse': np.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100,
        }

    def save_model(
        self,
        path: str = 'artifacts/trained_models/lightgbm_model.pkl',
    ):
        """Save model and feature list."""

        if self.model is None:
            raise ValueError("No model to save")

        joblib.dump(self.model, path)

        joblib.dump(
            self.feature_cols,
            'artifacts/feature_lists/lightgbm_features.pkl'
        )

        logger.info(f"Model saved to {path}")

    def load_model(
        self,
        path: str = 'artifacts/trained_models/lightgbm_model.pkl',
    ):
        """Load model and feature list."""

        self.model = joblib.load(path)

        self.feature_cols = joblib.load(
            'artifacts/feature_lists/lightgbm_features.pkl'
        )

        logger.info(f"Model loaded from {path}")
