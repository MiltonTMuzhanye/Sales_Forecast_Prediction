import pandas as pd
import numpy as np
from typing import Optional
from ..utils.logger import setup_logger
from ..utils.config import Config
from .lag_features import LagFeatureCreator
from .rolling_features import RollingFeatureCreator

logger = setup_logger(__name__)


class FeatureEngineer:
    """Handles feature engineering for sales forecasting."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self.target_col = self.config.get(
            'data.target_column',
            'Weekly_Sales'
        )
        self.date_col = self.config.get(
            'data.date_column',
            'Date'
        )

        self.group_cols = self.config.get(
            'data.features.group_columns',
            ['Store', 'Dept']
        )

        self.lags = self.config.get(
            'data.features.lags',
            [1, 2, 3, 4, 8, 12, 26, 52]
        )

        self.rolling_windows = self.config.get(
            'data.features.rolling_windows',
            [4, 8, 12, 26, 52]
        )

        self.rolling_stats = self.config.get(
            'data.features.rolling_stats',
            ['mean', 'std', 'min', 'max', 'median']
        )

        self.lag_creator = LagFeatureCreator(self.config)
        self.rolling_creator = RollingFeatureCreator(self.config)

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating time features...")
        df_copy = df.copy()

        if 'Month' in df_copy.columns:
            df_copy['month_sin'] = np.sin(
                2 * np.pi * df_copy['Month'] / 12
            )
            df_copy['month_cos'] = np.cos(
                2 * np.pi * df_copy['Month'] / 12
            )

        if 'DayOfWeek' in df_copy.columns:
            df_copy['dayofweek_sin'] = np.sin(
                2 * np.pi * df_copy['DayOfWeek'] / 7
            )
            df_copy['dayofweek_cos'] = np.cos(
                2 * np.pi * df_copy['DayOfWeek'] / 7
            )

            df_copy['is_weekend'] = (
                df_copy['DayOfWeek'] >= 5
            ).astype(int)

        if 'Quarter' in df_copy.columns:
            df_copy['quarter_sin'] = np.sin(
                2 * np.pi * df_copy['Quarter'] / 4
            )
            df_copy['quarter_cos'] = np.cos(
                2 * np.pi * df_copy['Quarter'] / 4
            )

        return df_copy

    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-based features."""
        logger.info("Creating store features...")
        df_copy = df.copy()

        if 'Type' in df_copy.columns:
            type_mapping = {'A': 0, 'B': 1, 'C': 2}
            df_copy['Type_encoded'] = df_copy['Type'].map(type_mapping)

        if 'Size' in df_copy.columns:
            df_copy['Size_log'] = np.log1p(df_copy['Size'])

        return df_copy

    def create_markdown_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create markdown-based features."""
        logger.info("Creating markdown features...")
        df_copy = df.copy()

        markdown_cols = self.config.get(
            'data.markdown_features',
            [
                'MarkDown1',
                'MarkDown2',
                'MarkDown3',
                'MarkDown4',
                'MarkDown5'
            ]
        )

        existing_markdowns = [
            col for col in markdown_cols
            if col in df_copy.columns
        ]

        if existing_markdowns:
            df_copy['MarkDown_Total'] = (
                df_copy[existing_markdowns].sum(axis=1)
            )
            df_copy['MarkDown_Avg'] = (
                df_copy[existing_markdowns].mean(axis=1)
            )
            df_copy['MarkDown_Count'] = (
                df_copy[existing_markdowns] > 0
            ).sum(axis=1)
            df_copy['MarkDown_Max'] = (
                df_copy[existing_markdowns].max(axis=1)
            )
            df_copy['MarkDown_Min'] = (
                df_copy[existing_markdowns].min(axis=1)
            )

        return df_copy

    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday-based features."""
        logger.info("Creating holiday features...")
        df_copy = df.copy()

        if 'IsHoliday' not in df_copy.columns:
            return df_copy

        if self.date_col in df_copy.columns:
            df_copy = (
                df_copy
                .sort_values(self.date_col)
                .reset_index(drop=True)
            )

        for i in [1, 2, 3]:
            df_copy[f'holiday_lag_{i}'] = (
                df_copy['IsHoliday']
                .shift(i)
                .fillna(0)
            )

        df_copy['holiday_window_3'] = (
            df_copy['IsHoliday']
            .rolling(3, min_periods=1)
            .sum()
            .fillna(0)
        )

        df_copy['holiday_window_5'] = (
            df_copy['IsHoliday']
            .rolling(5, min_periods=1)
            .sum()
            .fillna(0)
        )

        return df_copy

    def create_interaction_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features...")
        df_copy = df.copy()

        if (
            'Type_encoded' in df_copy.columns
            and 'IsHoliday' in df_copy.columns
        ):
            df_copy['Type_Holiday'] = (
                df_copy['Type_encoded'] * df_copy['IsHoliday']
            )

        if (
            'Temperature' in df_copy.columns
            and 'IsHoliday' in df_copy.columns
        ):
            df_copy['Temp_Holiday'] = (
                df_copy['Temperature'] * df_copy['IsHoliday']
            )

        if (
            'Fuel_Price' in df_copy.columns
            and 'IsHoliday' in df_copy.columns
        ):
            df_copy['Fuel_Holiday'] = (
                df_copy['Fuel_Price'] * df_copy['IsHoliday']
            )

        return df_copy

    def create_target_history_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create leakage-safe target-history features.

        Lag features use previous target values only.

        Rolling features use shift(1), so the current target is never
        included in its own rolling statistics.
        """
        logger.info("Creating target-history features...")

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' is required "
                "to create target-history features."
            )

        missing_groups = [
            col for col in self.group_cols
            if col not in df.columns
        ]

        if missing_groups:
            raise ValueError(
                f"Missing group columns required for target-history "
                f"features: {missing_groups}"
            )

        df_copy = self.lag_creator.create_lag_features(
            df.copy(),
            lags=self.lags,
            group_cols=self.group_cols
        )

        df_copy = self.rolling_creator.create_rolling_features(
            df_copy,
            windows=self.rolling_windows,
            stats=self.rolling_stats,
            group_cols=self.group_cols
        )

        logger.info(
            "Created %d lag/rolling target-history features.",
            len(self.lags) +
            (len(self.rolling_windows) * len(self.rolling_stats))
        )

        return df_copy

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_target_history: bool = False
    ) -> pd.DataFrame:
        """
        Run the complete feature-engineering pipeline.

        Parameters
        ----------
        df:
            Input dataframe.

        include_target_history:
            If True, create lag and rolling features derived from
            historical target values. This is disabled by default
            because future recursive forecasting requires special
            handling of these features.
        """
        logger.info("Running complete feature engineering pipeline...")

        df_copy = self.create_time_features(df)
        df_copy = self.create_store_features(df_copy)
        df_copy = self.create_markdown_features(df_copy)
        df_copy = self.create_holiday_features(df_copy)
        df_copy = self.create_interaction_features(df_copy)

        if include_target_history:
            df_copy = self.create_target_history_features(df_copy)

        logger.info(
            "Engineered dataset shape: %s",
            df_copy.shape
        )

        return df_copy
