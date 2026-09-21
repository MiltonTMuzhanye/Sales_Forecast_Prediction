import os
import joblib
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.exceptions import ModelTrainingError
from ..features.engineering import FeatureEngineer
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

logger = setup_logger(__name__)


class HybridModel:
    """Hybrid forecasting model combining Prophet and gradient boosting."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self.prophet_model = ProphetModel(self.config)
        self.feature_engineer = FeatureEngineer(self.config)

        self.ml_model = None
        self.use_xgboost = self.config.get(
            'model.models.hybrid.use_xgboost',
            True
        )

        self.target_col = self.config.get(
            'data.target_column',
            'Weekly_Sales'
        )

        self.date_col = self.config.get(
            'data.date_column',
            'Date'
        )

        self.trained = False

    def _get_prophet_fitted_values(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
    ) -> np.ndarray:
        """Generate in-sample Prophet predictions."""

        prophet_input = df[[date_col]].copy()
        prophet_input = prophet_input.rename(
            columns={date_col: 'ds'}
        )

        forecast = self.prophet_model.model.predict(
            prophet_input
        )

        return forecast['yhat'].to_numpy()

    def _build_ml_training_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        holiday_col: str,
    ) -> pd.DataFrame:
        """Create leakage-safe training data for the residual model."""

        working_df = df.copy()

        # Ensure chronological ordering.
        working_df[date_col] = pd.to_datetime(
            working_df[date_col]
        )
        working_df = working_df.sort_values(
            date_col
        ).reset_index(drop=True)

        # Generate the same engineered features used by the
        # tree-based forecasting models.
        working_df = self.feature_engineer.engineer_all_features(
            working_df
        )

        # Generate Prophet's fitted values for every historical row.
        prophet_fitted = self._get_prophet_fitted_values(
            working_df,
            date_col,
            target_col,
        )

        if len(prophet_fitted) != len(working_df):
            raise ValueError(
                "Prophet fitted predictions do not match "
                "the training data length."
            )

        working_df['prophet_prediction'] = prophet_fitted

        # Residual = actual - Prophet prediction.
        working_df['residual'] = (
            working_df[target_col]
            - working_df['prophet_prediction']
        )

        # IMPORTANT:
        # Weekly_Sales must not be supplied to the residual model.
        #
        # The residual model can use:
        # - historical/engineered explanatory features
        # - Prophet's prediction
        #
        # It must NOT use the actual target.
        return working_df

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'Weekly_Sales',
        date_col: str = 'Date',
        holiday_col: str = 'IsHoliday',
    ) -> None:
        """Train the Prophet + ML residual hybrid model."""

        logger.info("Training hybrid model...")

        try:
            if df.empty:
                raise ValueError(
                    "Cannot train hybrid model on empty dataframe."
                )

            required_columns = [
                date_col,
                target_col,
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            working_df = df.copy()

            working_df[date_col] = pd.to_datetime(
                working_df[date_col]
            )

            working_df = working_df.sort_values(
                date_col
            ).reset_index(drop=True)

            # ---------------------------------------------------------
            # STEP 1: Train Prophet
            # ---------------------------------------------------------
            logger.info(
                "Step 1/2: Training Prophet model..."
            )

            prophet_data = working_df[
                [date_col, target_col]
            ].copy()

            prophet_data.columns = ['ds', 'y']

            holidays_df = None

            if holiday_col in working_df.columns:
                holiday_rows = working_df[
                    working_df[holiday_col] == 1
                ][[date_col]].copy()

                if not holiday_rows.empty:
                    holiday_rows = holiday_rows.rename(
                        columns={date_col: 'ds'}
                    )

                    holiday_rows['holiday'] = (
                        'store_holiday'
                    )

                    holidays_df = holiday_rows

            self.prophet_model.train(
                prophet_data,
                holidays_df
            )

            logger.info(
                "Prophet model trained successfully."
            )

            # ---------------------------------------------------------
            # STEP 2: Build residual model
            # ---------------------------------------------------------
            logger.info(
                "Step 2/2: Training ML residual model..."
            )

            ml_data = self._build_ml_training_data(
                working_df,
                target_col,
                date_col,
                holiday_col,
            )

            if self.use_xgboost:
                logger.info(
                    "Using XGBoost for residual modelling."
                )
                self.ml_model = XGBoostModel(
                    self.config
                )
            else:
                logger.info(
                    "Using LightGBM for residual modelling."
                )
                self.ml_model = LightGBMModel(
                    self.config
                )

            # Train against residual.
            #
            # The updated XGBoost/LightGBM wrappers accept
            # target_col explicitly and use chronological splitting.
            #
            # Remove the actual target from the feature set by
            # creating a training dataframe without it.
            ml_training_data = ml_data.drop(
                columns=[target_col]
            )

            self.ml_model.train(
                ml_training_data,
                target_col='residual',
                test_size=0.2,
            )

            self.trained = True

            logger.info(
                "Hybrid model trained successfully."
            )

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train hybrid model: {e}"
            ) from e

    def _create_future_features(
        self,
        df: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Create ML features for future forecast periods."""

        date_col = self.date_col

        last_row = df.iloc[-1:].copy()

        future_df = pd.concat(
            [last_row] * len(future_dates),
            ignore_index=True,
        )

        future_df[date_col] = future_dates

        # Future holiday indicator cannot be inferred from the
        # last historical row. Default to 0 unless the dataset
        # explicitly supplies a future holiday schedule.
        if 'IsHoliday' in future_df.columns:
            future_df['IsHoliday'] = 0

        # Generate calendar and engineered features using
        # the same feature engineering pipeline used during training.
        future_df = self.feature_engineer.engineer_all_features(
            future_df
        )

        return future_df

    def predict(
        self,
        df: pd.DataFrame,
        periods: int = 12,
    ) -> np.ndarray:
        """Generate future hybrid forecasts."""

        if not self.trained:
            raise ValueError(
                "Model not trained."
            )

        if periods <= 0:
            raise ValueError(
                "periods must be greater than zero."
            )

        if df.empty:
            raise ValueError(
                "Cannot forecast from an empty dataframe."
            )

        logger.info(
            f"Making hybrid predictions for "
            f"{periods} periods..."
        )

        try:
            working_df = df.copy()

            working_df[self.date_col] = pd.to_datetime(
                working_df[self.date_col]
            )

            working_df = working_df.sort_values(
                self.date_col
            ).reset_index(drop=True)

            last_date = working_df[
                self.date_col
            ].max()

            # Weekly Walmart sales data.
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=periods,
                freq='7D',
            )

            # ---------------------------------------------------------
            # Prophet forecast
            # ---------------------------------------------------------
            prophet_forecast = self.prophet_model.predict(
                periods=periods
            )

            prophet_predictions = (
                prophet_forecast['yhat']
                .to_numpy()
            )

            if len(prophet_predictions) != periods:
                raise ValueError(
                    "Prophet prediction length does not "
                    "match requested forecast periods."
                )

            # ---------------------------------------------------------
            # ML residual forecast
            # ---------------------------------------------------------
            future_df = self._create_future_features(
                working_df,
                future_dates,
            )

            # Prophet prediction becomes an input feature
            # for the residual model.
            future_df['prophet_prediction'] = (
                prophet_predictions
            )

            residual_predictions = (
                self.ml_model.predict(future_df)
            )

            if len(residual_predictions) != periods:
                raise ValueError(
                    "Residual prediction length does not "
                    "match requested forecast periods."
                )

            # ---------------------------------------------------------
            # Final hybrid forecast
            # ---------------------------------------------------------
            final_predictions = (
                prophet_predictions
                + residual_predictions
            )

            return np.asarray(
                final_predictions
            )

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to make predictions: {e}"
            ) from e

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Evaluate hybrid model."""

        return {
            'rmse': np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),
            'mae': mean_absolute_error(
                y_true,
                y_pred
            ),
            'mape': mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100,
        }

    def save_model(
        self,
        path: str = 'artifacts/trained_models/hybrid_model/',
    ):
        """Save hybrid model."""

        if self.ml_model is None:
            raise ValueError(
                "No ML model available to save."
            )

        os.makedirs(
            path,
            exist_ok=True
        )

        self.prophet_model.save_model(
            os.path.join(
                path,
                'prophet_model.pkl'
            )
        )

        self.ml_model.save_model(
            os.path.join(
                path,
                'ml_model.pkl'
            )
        )

        joblib.dump(
            {
                'use_xgboost': self.use_xgboost,
                'trained': self.trained,
                'target_col': self.target_col,
                'date_col': self.date_col,
            },
            os.path.join(
                path,
                'config.pkl'
            ),
        )

        logger.info(
            f"Hybrid model saved to {path}"
        )

    def load_model(
        self,
        path: str = 'artifacts/trained_models/hybrid_model/',
    ):
        """Load hybrid model."""

        config_data = joblib.load(
            os.path.join(
                path,
                'config.pkl'
            )
        )

        self.use_xgboost = config_data[
            'use_xgboost'
        ]

        self.trained = config_data[
            'trained'
        ]

        self.target_col = config_data.get(
            'target_col',
            self.target_col
        )

        self.date_col = config_data.get(
            'date_col',
            self.date_col
        )

        self.prophet_model.load_model(
            os.path.join(
                path,
                'prophet_model.pkl'
            )
        )

        if self.use_xgboost:
            self.ml_model = XGBoostModel(
                self.config
            )
        else:
            self.ml_model = LightGBMModel(
                self.config
            )

        self.ml_model.load_model(
            os.path.join(
                path,
                'ml_model.pkl'
            )
        )

        logger.info(
            f"Hybrid model loaded from {path}"
        )
